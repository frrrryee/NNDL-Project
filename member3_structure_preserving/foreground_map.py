"""Foreground saliency maps for the repaired modulation.

Why this module exists
----------------------
The original method modulates the style strength with a soft Sobel edge
response ``M``:

    alpha = alpha0 * (1 - beta * M)

The intent is to protect the structure of the object.  The measurement in
Section IV.E shows that it does not: the spatial variant retains *less*
edge structure than a uniform control at the same effective strength
(-0.0031 on average, worse on 62.9% of the images), and the images that
lose the most are the ones with the lowest edge density.

The diagnosis is in the shape of the map.  ``M`` is a *texture* response,
not a *foreground* response:

  * in a textured background -- foliage, brickwork, canvas weave -- the
    gradient is large almost everywhere, so the normalised, dilated and
    blurred map saturates near 1 across the whole background;
  * on the smooth interior of the object the gradient is small, so the
    map is near 0.

With ``alpha = alpha0 (1 - beta M)`` that means the background receives
the *weak* style (0.15 at the main setting) and the object receives the
*full* style (0.5).  The styling damage is therefore concentrated on the
class cue, which is the opposite of the design.

This module provides maps that respond to the object rather than to the
texture, so that the same formula puts the weak style on the object.

The candidates
--------------
Three unsupervised maps are provided.  None of them uses a target-domain
image, a class label, or a learned segmentation model, and none of them
is fitted on the data: they are computed per image from that image alone.

``spectral``
    Spectral residual saliency: the log-amplitude spectrum of the image
    minus its local average, inverted and squared.  What survives this
    filter is the part of the image that is *distinctive* rather than
    *repetitive*, which is a much better proxy for "the object" than the
    gradient magnitude is.

``center``
    A Gaussian centre prior.  It is the crudest candidate and is included
    as the control: if the object is simply assumed to be in the middle,
    the method cannot fail for want of a foreground estimator, and the
    comparison shows how much the saliency term actually contributes.

``spectral_center``
    The spectral residual multiplied by the centre prior.  This is the
    combination that is expected to work best: the spectral term finds
    the distinctive region, the centre term breaks the tie in favour of
    the object when several regions are distinctive.

The selection between them is made by a pre-registered, target-free
criterion (see ``member4_training_eval/select_foreground_map.py``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ForegroundConfig:
    """The parameters of the foreground maps."""

    #: The size the image is reduced to before the spectrum is taken.
    spectral_size: int = 64
    #: The width of the Gaussian centre prior, as a fraction of the image.
    center_sigma: float = 0.35
    #: The weight of the centre prior in the combined map.
    center_weight: float = 0.5
    #: The blur applied to the final map, in pixels.
    smooth_sigma: float = 4.0

    def __post_init__(self):
        if not math.isfinite(self.spectral_size) or self.spectral_size < 8:
            raise ValueError("spectral_size must be finite and at least 8")
        for name in ("center_sigma", "center_weight", "smooth_sigma"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    def to_dict(self):
        return asdict(self)


#: The map kinds this module can build.
KINDS = ("spectral", "center", "spectral_center")


def _gaussian_kernel1d(sigma: float, device, dtype):
    radius = max(1, math.ceil(3 * sigma))
    axis = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (axis / max(sigma, 1e-6)).square())
    return kernel / kernel.sum()


def _blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur with replicate padding."""
    if sigma <= 0:
        return image
    kernel = _gaussian_kernel1d(sigma, image.device, image.dtype)
    radius = (kernel.numel() - 1) // 2
    channels = image.shape[1]
    horizontal = kernel.reshape(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.reshape(1, 1, -1, 1).expand(channels, 1, -1, 1)
    padded = F.pad(image, (radius, radius, 0, 0), mode="replicate")
    smoothed = F.conv2d(padded, horizontal, groups=channels)
    padded = F.pad(smoothed, (0, 0, radius, radius), mode="replicate")
    return F.conv2d(padded, vertical, groups=channels)


def _normalize(image: torch.Tensor) -> torch.Tensor:
    """Per-image normalisation to [0,1]."""
    flat = image.flatten(1)
    low = flat.min(dim=1).values.reshape(-1, 1, 1, 1)
    high = flat.max(dim=1).values.reshape(-1, 1, 1, 1)
    span = (high - low).clamp_min(1e-6)
    return ((image - low) / span).clamp(0, 1)


def _normalize_percentile(image: torch.Tensor, low_pct: float = 1.0,
                           high_pct: float = 99.0) -> torch.Tensor:
    """Per-image normalisation by percentiles, with the extremes clipped.

    The spectral residual is extremely peaked: a handful of pixels carry
    almost all of the energy, so a min/max normalisation maps everything
    else to a single flat value and the map becomes useless.  Percentile
    normalisation is the standard remedy in the saliency literature: the
    lower percentile becomes 0 and the upper one becomes 1, and the values
    outside that range are clipped.

    The clipping is what makes the map useful for the modulation, because
    it means the map is *saturated* over the object rather than
    continuously graded over the background.
    """
    flat = image.flatten(1)
    low = torch.quantile(flat, low_pct / 100.0, dim=1).reshape(-1, 1, 1, 1)
    high = torch.quantile(flat, high_pct / 100.0, dim=1).reshape(-1, 1, 1, 1)
    span = (high - low).clamp_min(1e-6)
    return ((image - low) / span).clamp(0, 1)


def spectral_saliency(image: torch.Tensor, config: ForegroundConfig) -> torch.Tensor:
    """The spectral residual saliency of an image.

    The image is reduced to a small grey-scale grid, its discrete Fourier
    transform is taken, the log amplitude of the transform is filtered by
    subtracting its local average, and the result is inverted and squared.
    The frequencies that are common to the whole image cancel; what
    remains is the image-specific structure.

    The result is a map in [0,1] where a large value means "this pixel is
    part of something distinctive in this particular image".
    """
    size = config.spectral_size
    weights = image.new_tensor([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1)
    grey = (image * weights).sum(dim=1, keepdim=True)
    small = F.interpolate(grey, (size, size), mode="bilinear",
                          align_corners=False)
    spectrum = torch.fft.fft2(small)
    log_amplitude = torch.log(spectrum.abs() + 1e-8)
    phase = torch.angle(spectrum)
    # A 3x3 circular box filter approximates the local average of the
    # log spectrum; it is what makes the residual "local".  Circular
    # padding keeps the filter continuous across the spectrum's wrap-around
    # boundary, which a zero pad would break.
    kernel = torch.ones(1, 1, 3, 3, device=image.device, dtype=image.dtype) / 9
    padded = F.pad(log_amplitude, (1, 1, 1, 1), mode="circular")
    local_average = F.conv2d(padded, kernel)
    residual = log_amplitude - local_average
    reconstructed = torch.fft.ifft2(torch.exp(residual) * torch.exp(1j * phase))
    saliency = (reconstructed.abs() ** 2).real
    saliency = F.interpolate(saliency, size=image.shape[-2:], mode="bilinear",
                             align_corners=False)
    return _normalize_percentile(saliency)


def center_prior(image: torch.Tensor, config: ForegroundConfig) -> torch.Tensor:
    """A Gaussian centre prior, normalised to [0,1]."""
    height, width = image.shape[-2:]
    sigma = max(1.0, config.center_sigma * min(height, width))
    ys = torch.arange(height, device=image.device, dtype=image.dtype) - (height - 1) / 2
    xs = torch.arange(width, device=image.device, dtype=image.dtype) - (width - 1) / 2
    grid_y = torch.exp(-0.5 * (ys / sigma).square())
    grid_x = torch.exp(-0.5 * (xs / sigma).square())
    prior = (grid_y[:, None] * grid_x[None, :]).reshape(1, 1, height, width)
    return prior.expand(image.shape[1], 1, height, width).contiguous()


def build_foreground_map(image: torch.Tensor, kind: str,
                         config: ForegroundConfig | None = None) -> torch.Tensor:
    """Build one of the foreground maps of an image.

    ``image`` is a B1HW or BCHW RGB tensor in [0,1].  The result is a
    B1HW map in [0,1] where a large value means "this pixel belongs to
    the object", i.e. the opposite convention to the edge map.
    """
    config = config or ForegroundConfig()
    if kind not in KINDS:
        raise ValueError(f"unknown foreground map: {kind}")
    image = torch.as_tensor(image).detach().float()
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[1] != 3 or min(image.shape) < 1:
        raise ValueError("image must be a non-empty RGB CHW or BCHW tensor")
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("image must contain finite RGB values in [0,1]")

    if kind == "spectral":
        result = spectral_saliency(image, config)
    elif kind == "center":
        result = center_prior(image, config)
    else:
        saliency = spectral_saliency(image, config)
        prior = center_prior(image, config)
        result = _normalize_percentile(saliency * (1.0 + config.center_weight * prior))
    return _blur(result, config.smooth_sigma).clamp(0, 1)
