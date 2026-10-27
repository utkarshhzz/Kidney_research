"""Re-implementation of the reference model CDR-VGG19 (Devi et al., ACISC 2025).

Paper description: contour-detected input -> VGG19 conv blocks -> FC(4096) ->
FC(4096) -> "normalized residual block" -> FC + softmax. The residual block
(paper Fig. 1) is Conv3x3-BN-ReLU -> Conv3x3-BN-ReLU -> Conv1x1-BN added to its
input. Because it sits *after* the fully connected layers it operates on a
4096-d vector, so the convolutions become linear layers. The paper reports
146M parameters (VGG19: 143M), i.e. ~3M for the block; a 4096-320-320-4096
bottleneck gives 2.7M and matches that budget.

Train it with ``data.input_mode=contour`` and ``data.cls_img_size=224`` to
reproduce the paper's setting under our leakage-free protocol.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision.models import VGG19_Weights, vgg19


class NormalizedResidualBlock(nn.Module):
    def __init__(self, dim: int = 4096, bottleneck: int = 320):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(dim, bottleneck), nn.BatchNorm1d(bottleneck), nn.ReLU(inplace=True),
            nn.Linear(bottleneck, bottleneck), nn.BatchNorm1d(bottleneck), nn.ReLU(inplace=True),
            nn.Linear(bottleneck, dim), nn.BatchNorm1d(dim),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.body(x))


class CDRVGG19(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        base = vgg19(weights=VGG19_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = base.features
        self.avgpool = base.avgpool
        self.fc = nn.Sequential(*list(base.classifier.children())[:-1])  # fc6, relu, drop, fc7, relu, drop
        self.residual = NormalizedResidualBlock(4096)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(4096, num_classes)
        self.cam_features = None

    def backbone_parameters(self):
        return self.features.parameters()

    def forward(self, x, task: str = "cls"):
        assert task == "cls"
        f = self.features(x)
        self.cam_features = f
        z = self.fc(self.avgpool(f).flatten(1))
        return {"logits": self.out(self.dropout(self.residual(z)))}
