"""Model construction and seeding shared by training and evaluation."""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

import config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class MixStyle(nn.Module):
    """MixStyle: mix the instance statistics of two feature maps.

    The statistics of the feature map are mixed between the current sample
    and another sample of the same batch::

        mu    = lambda * mu(x)   + (1 - lambda) * mu(x_shuffled)
        sigma = lambda * sigma(x) + (1 - lambda) * sigma(x_shuffled)

    and the feature map is re-normalised with the mixed statistics.  This is
    a domain-generalisation method: it does not generate images, it makes
    the network invariant to the style of the instance statistics, which is
    the same idea as the style augmentation of this project but applied in
    feature space instead of pixel space.

    The mixing is active during training only; at evaluation time the layer
    is an identity, so the reported accuracy is that of the plain network.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1,
                 eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0 or x.dim() != 4:
            return x

        # MixStyle is applied to the feature maps of the residual stages, so
        # the batch dimension is the only one that matters.
        batch = x.size(0)
        if batch < 2:
            return x

        # Beta(a, a) gives a symmetric distribution around 0.5, and the
        # lambda range [0, 1] is what keeps the mixed statistics inside the
        # convex hull of the two samples.
        lam = np.random.beta(self.alpha, self.alpha, size=batch)
        lam = torch.from_numpy(lam).to(x.device, dtype=x.dtype).view(batch, 1, 1, 1)

        perm = torch.randperm(batch, device=x.device)
        shuffled = x[perm]

        mean = x.mean(dim=[1, 2, 3], keepdim=True)
        std = (x.var(dim=[1, 2, 3], keepdim=True) + self.eps).sqrt()
        shuffled_mean = shuffled.mean(dim=[1, 2, 3], keepdim=True)
        shuffled_std = (shuffled.var(dim=[1, 2, 3], keepdim=True) + self.eps).sqrt()

        mixed_mean = lam * mean + (1 - lam) * shuffled_mean
        mixed_std = lam * std + (1 - lam) * shuffled_std

        normalized = (x - mean) / std
        return normalized * mixed_std + mixed_mean


class _StageWithMixStyle(nn.Module):
    """A residual stage preceded by a MixStyle layer.

    The stage's children are re-registered under this module with their
    original indices shifted by one, so the parameter names of the whole
    network stay exactly the ones of a plain torchvision ResNet
    (``layer1.0.bn1.weight`` and not ``layer1.stage.0.bn1.weight``).  A
    checkpoint of a MixStyle run therefore has the same key names as a
    checkpoint of a plain run.

    The forward pass reproduces the torchvision residual stage exactly: the
    shortcut is added to the output of the last block, and the downsample
    shortcut is applied when the stage has one.
    """

    def __init__(self, stage: nn.Module, p: float, alpha: float):
        super().__init__()
        self.mixstyle = MixStyle(p, alpha)
        self._blocks = list(stage.children())
        # The blocks keep their original indices, so the parameter names of
        # the network are exactly the ones of a plain torchvision ResNet.
        for index, block in enumerate(self._blocks):
            self.add_module(str(index), block)

    def forward(self, x):
        x = self.mixstyle(x)
        # Each torchvision block already adds its own shortcut, so the
        # stage is a plain chain of blocks once the mixing has been applied.
        for block in self._blocks:
            x = block(x)
        return x


class MixStyleResNet(nn.Module):
    """A ResNet-18 with a MixStyle layer in front of every residual stage."""

    def __init__(self, resnet: nn.Module, p: float = 0.5, alpha: float = 0.1):
        super().__init__()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = _StageWithMixStyle(resnet.layer1, p, alpha)
        self.layer2 = _StageWithMixStyle(resnet.layer2, p, alpha)
        self.layer3 = _StageWithMixStyle(resnet.layer3, p, alpha)
        self.layer4 = _StageWithMixStyle(resnet.layer4, p, alpha)
        self.avgpool = resnet.avgpool
        self.fc = resnet.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_model(arch: str = "resnet18", mixstyle: bool = False,
                mixstyle_p: float = 0.5, mixstyle_alpha: float = 0.1):
    """ImageNet-pretrained ResNet-18 with a 7-way head.

    ``mixstyle=False`` is the default and reproduces the architecture of
    every reported run exactly.  The MixStyle variant is a separate
    experimental group with its own fingerprint, because a changed
    architecture is a different model and must never be compared with the
    others as if it were the same one.
    """
    if arch != "resnet18":
        raise ValueError(f"unsupported architecture: {arch}")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if mixstyle:
        model = _insert_mixstyle(model, mixstyle_p, mixstyle_alpha)
    model.fc = nn.Linear(model.fc.in_features, len(config.CLASSES))
    return model.to(config.DEVICE)


def _insert_mixstyle(model: nn.Module, p: float, alpha: float) -> nn.Module:
    """Insert a MixStyle layer in front of every residual stage."""
    return MixStyleResNet(model, p, alpha)
