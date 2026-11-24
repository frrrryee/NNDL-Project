"""Objectness maps from an ImageNet-pretrained network's attention.

Why this module exists
----------------------
The repaired modulation (``ours_fg``) fixed the *sign* of the mechanism:
its EdgeScore gain over the matched control went from -0.0031 (the
Sobel map) to +0.0009.  But the amplitude stayed tiny, and the reason is
measurable: the spectral saliency map marks only 2-3% of the pixels as
foreground, so it protects the single most distinctive patch of the
object while the *rest* of the object still receives the full style
strength.

The cure is a better map, and the cheapest high-quality foreground
estimator available here is the one we already own: the ImageNet
pretrained ResNet-18 that the classifier is built from.  Its class
activation maps locate the object; taking the maximum over *all*
ImageNet classes turns them into a label-free objectness map.

Two properties make this legitimate:

  * the network is ImageNet-pretrained and PACS is not part of ImageNet,
    so no target-domain image influences the map;
  * the map is used only to *generate* the augmentation images.  The
    classifier is trained from the same ImageNet checkpoint as every
    other group and never sees the map.

The map is deliberately taken over all classes rather than over the
target class.  Using the label would make the augmentation a
label-dependent operation, and the label is exactly what the classifier
is supposed to infer.

How the map is built
--------------------
For one image:

  1. the ImageNet network is run forward and the feature map of the
     last convolutional stage is kept;
  2. the class weights of the final layer are used as 1x1 convolutions,
     which gives one activation map per ImageNet class;
  3. the maps are upsampled to the input resolution and the *maximum*
     over classes is taken, which is the objectness map;
  4. the map is normalised per image and blurred.

The result has the same convention as the other maps of this project:
a large value means "this pixel belongs to the object", so the
strength rule ``alpha = alpha0 (1 - beta M)`` puts the weak style on
the object.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import models

from member3_structure_preserving.foreground_map import (
    ForegroundConfig, _blur, _normalize_percentile,
)


@dataclass(frozen=True)
class AttentionConfig:
    """The parameters of the objectness map."""

    #: The side the image is reduced to before the network is run.
    input_size: int = 224
    #: The width of the final layer's receptive field in feature pixels.
    #: ResNet-18's layer4 has a stride of 32, so a 224-pixel input gives
    #: a 7x7 feature map.
    feature_stride: int = 32
    #: The blur applied to the map, in pixels.
    smooth_sigma: float = 4.0
    #: The percentile used for the normalisation.
    high_percentile: float = 99.0
    low_percentile: float = 1.0

    def __post_init__(self):
        if not math.isfinite(self.input_size) or self.input_size < 32:
            raise ValueError("input_size must be finite and at least 32")
        if not math.isfinite(self.smooth_sigma) or self.smooth_sigma < 0:
            raise ValueError("smooth_sigma must be finite and non-negative")

    def to_dict(self):
        return {"input_size": self.input_size,
                "feature_stride": self.feature_stride,
                "smooth_sigma": self.smooth_sigma,
                "high_percentile": self.high_percentile,
                "low_percentile": self.low_percentile}


class ObjectnessAttention:
    """The label-free objectness map of an ImageNet-pretrained ResNet-18.

    The network is loaded once and reused.  It is kept in evaluation
    mode and on the same device as the caller.
    """

    def __init__(self, config: AttentionConfig | None = None,
                 device: str | torch.device = "cpu"):
        self.config = config or AttentionConfig()
        self.device = torch.device(device)
        network = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # The class weights are kept: they are used as the 1x1
        # convolution that produces the activation maps.  The
        # classification head itself is dropped, because only the
        # convolutional trunk and the weights are needed.
        self.classifier = network.fc
        network.fc = torch.nn.Identity()
        self.trunk = network.to(self.device).eval()
        for parameter in self.trunk.parameters():
            parameter.requires_grad_(False)
        # The last residual stage is the one whose feature map the
        # activation maps are computed from.  A forward hook keeps it:
        # the classification head cannot return it, and the stage is
        # needed *before* the global pooling.
        self._features: list[torch.Tensor] = []
        self._hook = self.trunk.layer4.register_forward_hook(
            self._capture)

    def _capture(self, _module, _inputs, output):
        self._features = [output]
        return output

    @torch.no_grad()
    def maps(self, image: torch.Tensor) -> torch.Tensor:
        """The raw per-class activation maps of one image.

        ``image`` is a B3HW tensor in [0,1].  The result is a
        B(1 x 1000) tensor of maps at the input resolution.
        """
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must be a BCHW RGB tensor")
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("image must contain finite RGB values in [0,1]")
        size = self.config.input_size
        resized = F.interpolate(image, (size, size), mode="bilinear",
                                align_corners=False)
        # ImageNet normalisation: the pretrained network expects the
        # standard statistics.
        mean = resized.new_tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = resized.new_tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        normalised = (resized - mean) / std

        self._features = []
        self.trunk(normalised)
        if not self._features:
            raise RuntimeError("the trunk did not produce a feature map")
        features = self._features[0]
        if features.ndim != 4:
            raise RuntimeError("the feature map is not four-dimensional")
        weights = self.classifier.weight.to(device=features.device,
                                             dtype=features.dtype)
        # The class weights act as 1x1 convolutions over the feature map.
        activation = F.conv2d(features, weights.unsqueeze(-1).unsqueeze(-1))
        activation = F.relu(activation)
        activation = F.interpolate(activation, size=image.shape[-2:],
                                   mode="bilinear", align_corners=False)
        # The maximum over every ImageNet class: the map says "something
        # is here", not "this particular thing is here".
        return activation.max(dim=1, keepdim=True).values

    @torch.no_grad()
    def objectness(self, image: torch.Tensor) -> torch.Tensor:
        """The normalised objectness map of one image, in [0,1]."""
        raw = self.maps(image)
        normalised = _normalize_percentile(
            raw, low_pct=self.config.low_percentile,
            high_pct=self.config.high_percentile)
        return _blur(normalised, self.config.smooth_sigma).clamp(0, 1)

    def fingerprint(self) -> str:
        """A hash of the weights, so a run records which map produced it."""
        import hashlib
        digest = hashlib.sha256()
        for parameter in self.trunk.state_dict().values():
            digest.update(parameter.detach().cpu().numpy().tobytes())
        return digest.hexdigest()[:12]
