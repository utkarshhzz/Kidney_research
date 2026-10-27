"""KidneyMTL: one shared backbone, two task heads.

    input (tri-channel CT) -> backbone (timm, features at strides 4/8/16/32)
        |- stride-32 -> attention (CBAM) -> GAP -> MLP -> 4-class logits
        |- FPN neck (strides 32->4)      -> CenterNet head -> stone heatmap,
                                                              size, offset

Classification batches come from Dataset 2 and detection batches from
Dataset 1; the backbone is updated by both (partially-labelled multi-task
learning). With ``det_enabled=False`` it reduces to a single-task classifier
with the identical classification path (clean ablation).
"""
from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import build_activation, build_attention


def _create_backbone(name: str, pretrained: bool, drop_path_rate: float):
    probe = timm.create_model(name, pretrained=False, features_only=True)
    red = probe.feature_info.reduction()
    wanted = []
    for r in (4, 8, 16, 32):
        idx = [i for i, rr in enumerate(red) if rr == r]
        if not idx:
            raise ValueError(f"{name} has no feature map at stride {r} (strides: {red})")
        wanted.append(idx[-1])
    kwargs = dict(pretrained=pretrained, features_only=True, out_indices=tuple(wanted))
    try:
        return timm.create_model(name, drop_path_rate=drop_path_rate, **kwargs)
    except TypeError:
        return timm.create_model(name, **kwargs)


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3):
        super().__init__(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout),
                         nn.SiLU(inplace=True))


class FPNNeck(nn.Module):
    """Top-down FPN that fuses strides 32/16/8/4 into one stride-4 map."""

    def __init__(self, in_channels: list[int], ch: int):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, ch, 1) for c in in_channels])
        self.smooth = nn.ModuleList([ConvBNAct(ch, ch) for _ in in_channels[:-1]])

    def forward(self, feats):
        x = self.lateral[-1](feats[-1])
        for i in range(len(feats) - 2, -1, -1):
            x = F.interpolate(x, size=feats[i].shape[-2:], mode="nearest") + self.lateral[i](feats[i])
            x = self.smooth[i](x)
        return x


class CenterNetHead(nn.Module):
    def __init__(self, ch: int, num_classes: int = 1, prior: float = 0.1):
        super().__init__()

        def branch(cout):
            return nn.Sequential(ConvBNAct(ch, ch), nn.Conv2d(ch, cout, 1))

        self.hm = branch(num_classes)
        self.wh = branch(2)
        self.off = branch(2)
        nn.init.constant_(self.hm[-1].bias, -math.log((1 - prior) / prior))
        nn.init.constant_(self.wh[-1].bias, math.log(4.0))  # ~4 cells = 16 px at stride 4

    def forward(self, x):
        return {
            "hm": self.hm(x),                                   # logits
            "wh": torch.exp(self.wh(x).clamp(-4.0, 6.0)),       # box size in feature cells
            "off": torch.sigmoid(self.off(x)),                  # sub-cell centre offset
        }


class KidneyMTL(nn.Module):
    def __init__(self, backbone: str, num_classes: int, pretrained: bool = True, attention: str = "cbam",
                 cls_hidden: int = 256, cls_dropout: float = 0.3, head_activation: str = "silu",
                 drop_path_rate: float = 0.1, det_enabled: bool = True, neck_channels: int = 128,
                 det_stride: int = 4):
        super().__init__()
        self.backbone = _create_backbone(backbone, pretrained, drop_path_rate)
        chans = self.backbone.feature_info.channels()
        self.attn = build_attention(attention, chans[-1])
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(cls_dropout),
            nn.Linear(chans[-1], cls_hidden), build_activation(head_activation),
            nn.Dropout(cls_dropout), nn.Linear(cls_hidden, num_classes))
        self.det_enabled = det_enabled
        if det_enabled:
            assert det_stride == 4, "neck outputs stride 4"
            self.neck = FPNNeck(chans, neck_channels)
            self.det_head = CenterNetHead(neck_channels)
        self.cam_features = None

    def backbone_parameters(self):
        return self.backbone.parameters()

    def forward(self, x, task: str = "cls"):
        feats = self.backbone(x)
        if task == "cls":
            f = self.attn(feats[-1])
            self.cam_features = f
            return {"logits": self.cls_head(f)}
        if task == "det":
            if not self.det_enabled:
                raise RuntimeError("detection head disabled (model.det_enabled=false)")
            return self.det_head(self.neck(feats))
        if task == "both":
            f = self.attn(feats[-1])
            self.cam_features = f
            out = {"logits": self.cls_head(f)}
            if self.det_enabled:
                out.update(self.det_head(self.neck(feats)))
            return out
        raise ValueError(task)
