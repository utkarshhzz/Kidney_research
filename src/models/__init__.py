"""Model factory."""
from __future__ import annotations

import torch.nn as nn


def build_model(cfg) -> nn.Module:
    m = cfg.model
    num_classes = len(cfg.data.classes)
    if m.name == "kidneymtl":
        from .kidney_mtl import KidneyMTL

        return KidneyMTL(
            backbone=m.backbone, num_classes=num_classes, pretrained=m.pretrained, attention=m.attention,
            cls_hidden=m.cls_hidden, cls_dropout=m.cls_dropout, head_activation=m.head_activation,
            drop_path_rate=m.drop_path_rate, det_enabled=m.det_enabled, neck_channels=m.neck_channels,
            det_stride=m.det_stride)
    if m.name == "timm_classifier":
        from .base_models import TimmClassifier

        return TimmClassifier(m.backbone, num_classes, pretrained=m.pretrained, drop_rate=m.cls_dropout,
                              drop_path_rate=m.drop_path_rate, img_size=cfg.data.cls_img_size)
    if m.name == "cdr_vgg19":
        from .cdr_vgg19 import CDRVGG19

        return CDRVGG19(num_classes, pretrained=m.pretrained)
    raise ValueError(f"Unknown model.name '{m.name}'")


def has_detection(model: nn.Module) -> bool:
    return bool(getattr(model, "det_enabled", False))


def param_groups(model: nn.Module, backbone_lr: float, head_lr: float, weight_decay: float):
    """Lower LR for the pretrained backbone, no weight decay on norms/biases."""
    if hasattr(model, "backbone_parameters"):
        bb_ids = {id(p) for p in model.backbone_parameters()}
    else:  # timm classifier: everything except the classifier is backbone
        head = model.net.get_classifier()
        head_ids = {id(p) for p in head.parameters()}
        bb_ids = {id(p) for p in model.parameters() if id(p) not in head_ids}
    groups = {(lr, wd): [] for lr in (backbone_lr, head_lr) for wd in (weight_decay, 0.0)}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        lr = backbone_lr if id(p) in bb_ids else head_lr
        wd = 0.0 if p.ndim <= 1 else weight_decay
        groups[(lr, wd)].append(p)
    return [{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in groups.items() if ps]
