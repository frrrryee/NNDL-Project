"""Line-art rendering with an attention gate and a colour-block vote.

Why this module exists
----------------------
The repaired modulation (``ours_fg``) fixed the *sign* of the mechanism:
its EdgeScore gain over the matched control went from -0.0031 (the Sobel
map) to +0.0009.  The amplitude stayed tiny, and the reason is
measurable: the spectral saliency map marks only 2-3% of the pixels as
foreground, so it protects the single most distinctive patch of the
object while the rest of the object still receives the full style
strength.

This module produces a *line drawing* of the content image instead of a
stylised image.  It attacks a different hypothesis -- shape bias -- and it
does so with two signals that fail in different ways:

``attention``
    A semantic signal.  It knows where the object is, but its boundary is
    coarse: the CAM of a ResNet-18 lives on a 7x7 grid.

``colour blocks``
    A geometric signal.  Superpixels follow the image's own colour
    structure, so their boundaries are pixel-accurate, but they know
    nothing about semantics: a colour edge inside the background is
    exactly as strong as the object's outline.

Used alone, the colour blocks degenerate into an edge detector -- which
is the very thing that failed in the original method.  Used alone, the
attention is a blurry blob.  Together they are complementary, and the
combination is what this module implements.

The combination
----------------
The two signals are combined at the *region* level, not at the pixel
level:

1. the image is cut into superpixels;
2. each superpixel receives the mean attention of its pixels;
3. the drawing is placed on the *boundaries* between superpixels, and
   the darkness of a boundary segment is decided by the attention of the
   regions on either side of it.

So the colour blocks answer "where are the boundaries" and the attention
answers "is this boundary worth drawing".  The background is not deleted:
its boundaries are drawn faintly.  That keeps the context a classifier
may need -- a horse on grass and a horse in a stable are different
pictures -- while the object's outline dominates.

Two artefacts are produced for every image:

``ours``
    The gated line drawing.

``plain``
    The same line drawing with a uniform gate, i.e. every boundary drawn
    at the same darkness.  This is the strength-matched control: it
    isolates the effect of the gate from the effect of drawing fewer
    lines.

Nothing here is fitted on data and no label is used: the attention comes
from an ImageNet-pretrained network and the superpixels from the image
itself.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LineArtConfig:
    """The parameters of the line-art rendering."""

    #: The number of superpixels.
    n_segments: int = 400
    #: The compactness of the superpixel segmentation.
    compactness: float = 12.0
    #: The number of superpixels to draw.
    line_width: int = 2
    #: The darkness of a boundary between two regions that both carry
    #: attention.  A boundary between two background regions is drawn at
    #: ``background_strength`` times this.
    foreground_strength: float = 1.0
    background_strength: float = 0.35
    #: The attention above which a region counts as foreground.
    foreground_threshold: float = 0.5
    #: The side the attention map is computed at.
    attention_size: int = 224
    #: The blur applied to the attention map, in pixels.
    smooth_sigma: float = 4.0

    def __post_init__(self):
        if self.n_segments < 50:
            raise ValueError("n_segments must be at least 50")
        if not math.isfinite(self.compactness) or self.compactness <= 0:
            raise ValueError("compactness must be finite and positive")
        for name in ("foreground_strength", "background_strength",
                     "foreground_threshold", "smooth_sigma"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.background_strength > self.foreground_strength:
            raise ValueError("the background must not be drawn darker than "
                             "the foreground")

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return image
    radius = max(1, math.ceil(3 * sigma))
    axis = torch.arange(-radius, radius + 1, device=image.device,
                        dtype=image.dtype)
    kernel = torch.exp(-0.5 * (axis / max(sigma, 1e-6)).square())
    kernel = kernel / kernel.sum()
    channels = image.shape[1]
    horizontal = kernel.reshape(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.reshape(1, 1, -1, 1).expand(channels, 1, -1, 1)
    padded = F.pad(image, (radius, radius, 0, 0), mode="replicate")
    smoothed = F.conv2d(padded, horizontal, groups=channels)
    padded = F.pad(smoothed, (0, 0, radius, radius), mode="replicate")
    return F.conv2d(padded, vertical, groups=channels)


def superpixel_labels(image: np.ndarray, config: LineArtConfig) -> np.ndarray:
    """The superpixel label of every pixel.

    ``image`` is an HxWx3 uint8 array.  The result is an HxW int array.
    SLIC is used because it is the standard superpixel method and it is
    fast; the alternative is the Felzenszwalb segmentation, which is
    faster still but gives larger and less regular regions.
    """
    try:
        from skimage.segmentation import slic
    except ImportError as error:  # pragma: no cover
        raise ImportError("scikit-image is required for the superpixels") \
            from error
    rgb = image.astype(np.float32) / 255.0
    labels = slic(rgb, n_segments=config.n_segments,
                  compactness=config.compactness, start_label=0,
                  channel_axis=-1)
    return labels.astype(np.int32)


def region_attention(labels: np.ndarray, attention: np.ndarray) -> np.ndarray:
    """The mean attention of every superpixel.

    ``attention`` is an HxW float array in [0,1].  The result is a
    1-D array with one value per label, indexed by the label.
    """
    count = int(labels.max()) + 1
    total = np.bincount(labels.ravel(), weights=attention.ravel(),
                        minlength=count)
    size = np.bincount(labels.ravel(), minlength=count)
    return total / np.maximum(size, 1)


def boundary_map(labels: np.ndarray) -> np.ndarray:
    """A map that is 1 on a pixel adjacent to a different label."""
    height, width = labels.shape
    boundary = np.zeros((height, width), dtype=bool)
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    return boundary


def draw_line_art(image: np.ndarray, attention: np.ndarray,
                  config: LineArtConfig, gate: bool = True,
                  foreground_weight: float | None = None,
                  background_weight: float | None = None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render the line drawing of an image.

    ``image`` is an HxWx3 uint8 array and ``attention`` an HxW float
    array in [0,1].  The results are the drawing, the mask of the drawn
    pixels, and the *strength map* -- the darkness of each drawn pixel
    before it is turned into ink.

    The strength map is returned because the ink mass has to be measured
    to build the strength-matched control, and the darkness of a faint
    line cannot be recovered from the finished picture.

    When ``gate`` is false every boundary is drawn at the same darkness,
    which is the ungated reference.

    ``foreground_weight`` and ``background_weight`` override the
    strength of the two kinds of boundary.  They are *multipliers* on
    the configured strengths, so a weight of one reproduces the
    reference drawing and a weight of two doubles the ink of that
    kind of boundary.  They exist so that the matched controls can hold
    the ink mass fixed *by changing the relative weight of the object's
    outline against the background lines*, rather than by scaling the
    whole drawing: a uniform scaling would change the foreground ink as
    a side effect, and the two controls would then be the same image.
    """
    labels = superpixel_labels(image, config)
    attention = np.clip(attention, 0.0, 1.0)
    region = region_attention(labels, attention)

    height, width = labels.shape
    drawing = np.zeros((height, width), dtype=np.float32)

    # For every boundary pixel, look at the two regions that meet there
    # and decide how dark the line is.  The decision is taken from the
    # region values on both sides, so a boundary is only as strong as
    # the stronger of its two sides: the object's outline is strong on
    # one side, and so is a boundary between two parts of the object.
    boundary = boundary_map(labels)
    ys, xs = np.nonzero(boundary)
    if len(ys):
        left = labels[ys, np.maximum(xs - 1, 0)]
        right = labels[ys, np.minimum(xs + 1, width - 1)]
        up = labels[np.maximum(ys - 1, 0), xs]
        down = labels[np.minimum(ys + 1, height - 1), xs]
        side = np.maximum(np.maximum(region[left], region[right]),
                          np.maximum(region[up], region[down]))
        is_foreground = side >= config.foreground_threshold

        if gate:
            # The gated drawing: the object's outline is drawn at full
            # strength and the background lines fade out with the
            # attention.
            strength = np.where(
                is_foreground, config.foreground_strength,
                config.background_strength * side)
        else:
            # The ungated reference: every boundary is drawn at the same
            # darkness, unless the caller overrides the two weights.
            # The weights are multipliers on the configured strengths.
            foreground = config.foreground_strength * (
                1.0 if foreground_weight is None else foreground_weight)
            background = config.background_strength * (
                1.0 if background_weight is None else background_weight)
            strength = np.where(is_foreground, foreground, background)
        drawing[ys, xs] = np.clip(strength, 0.0, 1.0)

    # Dilate the boundary to the requested line width.
    if config.line_width > 1:
        dilated = drawing.copy()
        for shift in range(1, config.line_width):
            dilated[:, shift:] = np.maximum(dilated[:, shift:],
                                            drawing[:, :-shift])
            dilated[:, :-shift] = np.maximum(dilated[:, :-shift],
                                             drawing[:, shift:])
            dilated[shift:, :] = np.maximum(dilated[shift:, :],
                                            drawing[:-shift, :])
            dilated[:-shift, :] = np.maximum(dilated[:-shift, :],
                                             drawing[shift:, :])
        drawing = dilated

    mask = drawing > 0.05
    picture = np.full_like(image, 255)
    # The ink's darkness *is* the strength: the picture is white paper
    # with the line drawn on it, and the line is as dark as its
    # strength says.  This keeps the saved picture and the strength map
    # in exact correspondence, so the ink mass of a drawing can be
    # recovered from the picture alone.
    #
    # The obvious alternative -- darkening the source colour instead --
    # would make the darkness depend on the colour of the object as
    # well as on the strength, and the two ink-matched controls are
    # matched on the strength.  The two quantities would then differ,
    # and the controls would be matched on something the training never
    # sees.
    ink = 255.0 * (1.0 - drawing)
    picture[mask] = np.clip(ink[mask][:, None].repeat(3, axis=1),
                            0, 255).astype(np.uint8)
    return picture, mask, drawing

def attention_from_map(attention: torch.Tensor, height: int, width: int,
                       sigma: float) -> np.ndarray:
    """The attention as an HxW numpy array in [0,1]."""
    array = attention.detach().cpu()
    if array.ndim == 4:
        array = array[0, 0]
    elif array.ndim == 3:
        array = array[0]
    array = F.interpolate(array.reshape(1, 1, *array.shape[-2:]),
                          size=(height, width), mode="bilinear",
                          align_corners=False)
    array = _blur(array, sigma).clamp(0, 1)
    return array.reshape(height, width).numpy()


# ---------------------------------------------------------------------------
# The strength-matched controls
# ---------------------------------------------------------------------------
#
# The gated drawing and the uniform drawing do not carry the same amount
# of ink: the gate fades the background lines out, so the gated drawing
# has less ink in total but more in the foreground.  A control that
# differs in ink mass confounds "the gate helps" with "less ink is
# easier to learn", so the control has to be matched on ink.
#
# Two matchings are produced, because they exclude different confounds:
#
# ``total``
#     The total ink mass is equal.  This excludes "less ink is easier to
#     learn" outright -- including the possibility that the gate helps
#     merely by thinning the image out.
#
# ``foreground``
#     The ink mass inside the object is equal.  The background lines are
#     free to differ.  This excludes the narrower confound "the gate
#     helped by drawing the object's outline more heavily", which is the
#     confound the gate is *meant* to introduce.
#
# Neither matching is the "right" one on its own: the total matching
# equalises something the gate deliberately changes, and the foreground
# matching lets a difference in background ink through.  Reporting both
# makes the gate's effect interpretable either way.

def ink_masses(drawing: np.ndarray, mask: np.ndarray,
               foreground: np.ndarray) -> tuple[float, float]:
    """The total ink mass and the ink mass inside the object.

    ``drawing`` is the HxW strength map, ``mask`` the pixels that were
    drawn on, and ``foreground`` a boolean map of the object.  The mass
    of a pixel is its strength: a faint line contributes less than a
    dark one.
    """
    total = float(drawing[mask].sum())
    inside = float(drawing[mask & foreground].sum())
    return total, inside


def solve_ink_weights(drawing: np.ndarray, mask: np.ndarray,
                      foreground: np.ndarray, target_total: float | None = None,
                      target_foreground: float | None = None
                      ) -> tuple[float, float]:
    """The foreground and background weights that match a target ink mass.

    The matched control has to hold the ink mass fixed *without*
    changing the relative weight of the object's outline against the
    background lines, because that relative weight is the very thing
    the gate changes.  A uniform rescaling would change it, so the
    control is built by solving for the two weights instead.

    With ``target_total`` both weights are scaled by the same factor,
    so the total ink matches and the foreground-to-background ratio is
    the control's own.  With ``target_foreground`` only the foreground
    weight is adjusted and the background keeps its weight, so the
    object's outline carries the same ink as in the gated drawing while
    the background lines are free to differ.

    The two weights are returned rather than the drawing, because the
    drawing is produced by the caller, which knows how the weights
    enter the rendering.  They are multipliers on the configured
    strengths.
    """
    inside_mask = mask & foreground
    outside_mask = mask & ~foreground
    # The ink that the foreground lines and the background lines carry
    # in the drawing as it stands.
    foreground_mass = float(drawing[inside_mask].sum())
    background_mass = float(drawing[outside_mask].sum())

    if target_total is not None:
        if foreground_mass + background_mass <= 1e-6:
            return 1.0, 1.0
        factor = target_total / (foreground_mass + background_mass)
        return factor, factor

    if target_foreground is not None:
        if foreground_mass <= 1e-6:
            return 1.0, 1.0
        factor = target_foreground / foreground_mass
        return factor, 1.0

    return 1.0, 1.0


def rescale_ink(drawing: np.ndarray, mask: np.ndarray,
                foreground: np.ndarray, target_total: float | None = None,
                target_foreground: float | None = None) -> np.ndarray:
    """Rescale a drawing so that its ink mass matches a target.

    With ``target_total`` the whole drawing is scaled by one factor, so
    both the object's outline and the background lines keep their
    relative weights.  With ``target_foreground`` only the object's
    outline is scaled and the background lines are left alone.

    The scale factor is clipped to a sane range: an extreme factor would
    make the control either invisible or solid black, and neither would
    be a meaningful control.

    This function is the *uniform* rescale.  The matched controls of the
    line-art family are built with :func:`solve_ink_weights` instead,
    because a uniform rescale moves the foreground ink as a side effect.
    """
    if target_total is None and target_foreground is None:
        return drawing
    total, inside = ink_masses(drawing, mask, foreground)
    if target_total is not None:
        if total <= 1e-6:
            return drawing
        factor = target_total / total
    else:
        if inside <= 1e-6:
            return drawing
        factor = target_foreground / inside
    factor = float(np.clip(factor, 0.05, 20.0))
    scaled = drawing * factor
    return np.clip(scaled, 0.0, 1.0)


def foreground_mask(attention: np.ndarray,
                    config: LineArtConfig) -> np.ndarray:
    """The pixels that count as the object.

    The object is where the attention is above the threshold, smoothed
    and then closed, so that a hole inside the object does not split it.
    """
    binary = (attention >= config.foreground_threshold)
    if not binary.any():
        return binary
    try:
        from scipy.ndimage import binary_closing, binary_dilation
    except ImportError:  # pragma: no cover
        return binary
    radius = max(1, int(round(config.smooth_sigma)))
    closed = binary_closing(binary, iterations=2)
    grown = binary_dilation(closed, iterations=max(1, radius // 4))
    return grown
