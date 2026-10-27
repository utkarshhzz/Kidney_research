"""Single-task baseline classifiers (any timm model), same interface as KidneyMTL."""
from __future__ import annotations

import timm
import torch.nn as nn


class TimmClassifier(nn.Module):
    """Plain fine-tuned ImageNet backbone + linear classifier.

    Baselines used in the paper: resnet50, densenet121, efficientnet_b3,
    convnext_tiny, vit_small_patch16_224, vgg19_bn.
    """

    def __init__(self, backbone: str, num_classes: int, pretrained: bool = True,
                 drop_rate: float = 0.3, drop_path_rate: float = 0.1, img_size: int | None = None):
        super().__init__()
        kwargs = dict(pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
        if any(k in backbone for k in ("vit", "deit")) and img_size:
            kwargs["img_size"] = img_size
        try:
            self.net = timm.create_model(backbone, drop_path_rate=drop_path_rate, **kwargs)
        except TypeError:  # e.g. VGG has no stochastic depth
            self.net = timm.create_model(backbone, **kwargs)
        self.cam_features = None  # last spatial feature map (for Grad-CAM)

    def forward(self, x, task: str = "cls"):
        assert task == "cls", "baseline classifiers have no detection head"
        f = self.net.forward_features(x)
        self.cam_features = f  # (B,C,H,W) or ViT tokens (B,N,C); see to_spatial()
        return {"logits": self.net.forward_head(f)}

    def to_spatial(self, t):
        """ViT tokens (B,N,C) -> (B,C,s,s) without prefix tokens; CNN maps pass through."""
        if t.dim() != 3:
            return t
        tokens = t[:, getattr(self.net, "num_prefix_tokens", 1):]
        s = int(tokens.shape[1] ** 0.5)
        return tokens.transpose(1, 2).reshape(t.shape[0], -1, s, s)
