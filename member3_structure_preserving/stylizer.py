"""Member 3 generator, built on member 2's unchanged AdaIN encoder/decoder."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F

from member2_adain.adain import AdaINModel, tensor_to_pil
from member3_structure_preserving.foreground_map import (
    ForegroundConfig, build_foreground_map,
)
from member3_structure_preserving.structure_map import (
    StructureConfig, build_structure_map, spatial_alpha,
)


@dataclass
class PreparedContent:
    image: torch.Tensor
    features: torch.Tensor
    structure: torch.Tensor
    #: The foreground map, when the foreground mode is active.
    foreground: torch.Tensor | None = None


@dataclass
class StylizationResult:
    image: Image.Image
    content: Image.Image
    structure: torch.Tensor
    alpha_map: torch.Tensor
    feature_alpha: torch.Tensor
    baseline: Image.Image | None = None
    matched_baseline: Image.Image | None = None
    matched_alpha: float | None = None
    #: The map that produced ``alpha_map``: the edge map or the foreground map.
    modulation_map: torch.Tensor | None = None


class StructurePreservingStylizer:
    """Generate the styled images.

    ``modulation`` selects what modulates the style strength:

    ``edge``
        The original method.  The map is a soft Sobel edge response, and
        ``alpha = alpha0 (1 - beta M)`` reduces the strength where the
        gradient is large.

    ``foreground``
        The repaired method.  The map is an unsupervised foreground
        estimate, and the *same* formula reduces the strength where the
        map is large -- which is now the object rather than the texture.
        The formula is unchanged; only the map is different, so the
        comparison between the two isolates the map.
    """

    def __init__(self, model: AdaINModel | None = None,
                 structure_config: StructureConfig | None = None,
                 foreground_config: ForegroundConfig | None = None,
                 modulation: str = "edge",
                 foreground_kind: str = "spectral_center",
                 **model_options):
        self.model = model if model is not None else AdaINModel(**model_options)
        self.structure_config = structure_config or StructureConfig()
        self.foreground_config = foreground_config or ForegroundConfig()
        if modulation not in ("edge", "foreground"):
            raise ValueError(f"unknown modulation: {modulation}")
        self.modulation = modulation
        self.foreground_kind = foreground_kind
        self._style_cache: dict[Path, torch.Tensor] = {}

    @torch.no_grad()
    def prepare_content(self, content) -> PreparedContent:
        image = self.model.prepare_image(content, self.model.content_size).to(self.model.device)
        self._validate_image(image)
        structure = build_structure_map(image, self.structure_config)
        foreground = None
        if self.modulation == "foreground":
            foreground = build_foreground_map(image, self.foreground_kind,
                                             self.foreground_config)
        return PreparedContent(image, self.model.encode(image), structure, foreground)

    @staticmethod
    def _validate_image(image):
        if image.shape[0] != 1 or min(image.shape[-2:]) < 32:
            raise ValueError("stylization requires one RGB image, each side at least 32 pixels")
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("image values must be finite and in [0,1]")

    @torch.no_grad()
    def encode_style(self, style) -> torch.Tensor:
        key = Path(style).resolve() if isinstance(style, (str, Path)) else None
        if key is not None and key in self._style_cache:
            return self._style_cache[key]
        image = self.model.prepare_image(style, self.model.style_size).to(self.model.device)
        self._validate_image(image)
        features = self.model.encode(image)
        if key is not None:
            self._style_cache[key] = features
        return features

    @torch.no_grad()
    def render(self, content: PreparedContent, style_features: torch.Tensor,
               alpha: float = 0.5, beta: float = 0.7,
               with_baseline: bool = False,
               with_matched: bool = False) -> StylizationResult:
        # The map that modulates the strength.  The formula is the same in
        # both modes: the strength falls where the map is large.  Only the
        # map differs, and that is what the comparison isolates.
        modulation_map = (content.foreground if self.modulation == "foreground"
                          else content.structure)
        alpha_map = spatial_alpha(modulation_map, alpha, beta)
        # Area averaging preserves the contribution of narrow contours when
        # reducing the map to relu4_1 resolution. The equation is linear in M.
        feature_alpha = F.interpolate(alpha_map, size=content.features.shape[-2:], mode="area")
        mixed = self.model.mix(content.features, style_features, feature_alpha)
        image = tensor_to_pil(self.model.decode(mixed))
        baseline = None
        matched = None
        matched_alpha = None
        if with_baseline:
            baseline = tensor_to_pil(self.model.decode(
                self.model.mix(content.features, style_features, alpha)))
        if with_matched:
            # Strength-matched uniform AdaIN control.
            #
            # alpha(x, y) = alpha0 * (1 - beta * M) is smaller than alpha0
            # wherever M is large, so the spatial map applies *less* total
            # style than a uniform alpha0 mix.  Comparing the spatial map
            # against a uniform alpha0 AdaIN therefore confounds "structure
            # preservation" with "weaker stylization".  This control removes
            # that confound: it is a plain uniform AdaIN whose strength is
            # exactly the mean of the spatial alpha map, i.e. the same total
            # style energy, but with no spatial modulation at all.
            #
            # The mean is taken on the feature-resolution map that the mixer
            # actually uses, so the two images are matched on the quantity
            # that enters the objective.  The mean is computed on the CPU
            # copy so that the recorded value is exactly the mean of the
            # recorded map, independent of GPU reduction order.
            feature_alpha_cpu = feature_alpha.cpu()
            matched_alpha = float(feature_alpha_cpu.mean())
            matched = tensor_to_pil(self.model.decode(
                self.model.mix(content.features, style_features, matched_alpha)))
        return StylizationResult(image, tensor_to_pil(content.image),
                                 content.structure.cpu(), alpha_map.cpu(),
                                 feature_alpha.cpu(), baseline, matched, matched_alpha,
                                 modulation_map.cpu() if modulation_map is not None else None)

    def stylize(self, content, style, alpha=0.5, beta=0.7, with_baseline=False,
                with_matched=False):
        return self.render(self.prepare_content(content), self.encode_style(style),
                           alpha, beta, with_baseline, with_matched)


def stylize(content, style, alpha=0.5, beta=0.7,
            generator: StructurePreservingStylizer | None = None) -> Image.Image:
    """Single-image entry point; reuse a generator instance for many images."""
    generator = generator or StructurePreservingStylizer()
    return generator.stylize(content, style, alpha, beta).image
