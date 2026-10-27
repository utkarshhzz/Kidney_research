"""Loss functions: classification, CenterNet detection, multi-task weighting."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# classification
# ----------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., ICCV 2017) with class weights + label smoothing."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)
        self.ls = label_smoothing

    def forward(self, logits, target):
        logits = logits.float()
        logp = F.log_softmax(logits, dim=1)
        n = logits.shape[1]
        with torch.no_grad():
            t = torch.full_like(logp, self.ls / n)
            t.scatter_(1, target[:, None], 1.0 - self.ls + self.ls / n)
        pt = logp.gather(1, target[:, None]).exp().squeeze(1)
        ce = -(t * logp).sum(1)
        loss = (1 - pt).pow(self.gamma) * ce
        if self.weight is not None:
            w = self.weight[target]
            return (loss * w).sum() / w.sum()
        return loss.mean()


def build_cls_loss(cfg, weights: torch.Tensor | None) -> nn.Module:
    if cfg.loss.cls == "focal":
        return FocalLoss(cfg.loss.focal_gamma, weights, cfg.loss.label_smoothing)
    if cfg.loss.cls == "ce":
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.loss.label_smoothing)
    raise ValueError(cfg.loss.cls)


# ----------------------------------------------------------------------------
# box IoU losses
# ----------------------------------------------------------------------------
def box_iou_loss(pred, gt, kind: str = "ciou", eps: float = 1e-7):
    """pred, gt: (N,4) xyxy. Returns per-box loss 1 - (G|C)IoU (Zheng et al., AAAI 2020)."""
    px1, py1, px2, py2 = pred.unbind(-1)
    gx1, gy1, gx2, gy2 = gt.unbind(-1)
    iw = (torch.min(px2, gx2) - torch.max(px1, gx1)).clamp(min=0)
    ih = (torch.min(py2, gy2) - torch.max(py1, gy1)).clamp(min=0)
    inter = iw * ih
    pw, ph = (px2 - px1).clamp(min=eps), (py2 - py1).clamp(min=eps)
    gw, gh = (gx2 - gx1).clamp(min=eps), (gy2 - gy1).clamp(min=eps)
    union = pw * ph + gw * gh - inter + eps
    iou = inter / union
    cw = torch.max(px2, gx2) - torch.min(px1, gx1)
    ch = torch.max(py2, gy2) - torch.min(py1, gy1)
    if kind == "giou":
        c_area = cw * ch + eps
        return 1 - (iou - (c_area - union) / c_area)
    if kind == "ciou":
        c2 = cw ** 2 + ch ** 2 + eps
        rho2 = ((px1 + px2 - gx1 - gx2) ** 2 + (py1 + py2 - gy1 - gy2) ** 2) / 4
        v = (4 / math.pi ** 2) * (torch.atan(gw / gh) - torch.atan(pw / ph)) ** 2
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)
        return 1 - (iou - rho2 / c2 - alpha * v)
    raise ValueError(kind)


# ----------------------------------------------------------------------------
# CenterNet
# ----------------------------------------------------------------------------
def heatmap_focal_loss(logits, gt, alpha: float = 2.0, beta: float = 4.0):
    """Penalty-reduced pixel-wise focal loss (Zhou et al., 2019)."""
    pred = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = 1 - pos
    pos_loss = torch.log(pred) * (1 - pred).pow(alpha) * pos
    neg_loss = torch.log(1 - pred) * pred.pow(alpha) * (1 - gt).pow(beta) * neg
    num_pos = pos.sum()
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos.clamp(min=1.0)


def _gather(feat, ind):
    """feat (B,C,H,W), ind (B,K) -> (B,K,C)."""
    b, c = feat.shape[:2]
    feat = feat.view(b, c, -1).permute(0, 2, 1)
    return feat.gather(1, ind[..., None].expand(-1, -1, c))


class CenterNetLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w_hm = cfg.loss.det_hm_weight
        self.w_off = cfg.loss.det_off_weight
        self.w_box = cfg.loss.det_box_weight
        self.box_kind = cfg.loss.det_box_loss
        self.stride = cfg.model.det_stride

    def forward(self, out, batch):
        hm_loss = heatmap_focal_loss(out["hm"], batch["hm"])
        mask = batch["mask"]
        n = mask.sum().clamp(min=1.0)
        ind = batch["ind"]
        W = out["hm"].shape[-1]
        off = _gather(out["off"].float(), ind)
        wh = _gather(out["wh"].float(), ind)
        off_loss = (F.l1_loss(off, batch["off"], reduction="none").sum(-1) * mask).sum() / n

        cx = (ind % W).float() + off[..., 0]
        cy = (ind // W).float() + off[..., 1]
        s = self.stride
        pred = torch.stack([(cx - wh[..., 0] / 2) * s, (cy - wh[..., 1] / 2) * s,
                            (cx + wh[..., 0] / 2) * s, (cy + wh[..., 1] / 2) * s], -1)
        gt = batch["boxes"]
        m = mask.bool()
        if m.any():
            if self.box_kind == "l1":
                box_loss = F.l1_loss(pred[m] / s, gt[m] / s, reduction="sum") / n
            else:
                box_loss = box_iou_loss(pred[m], gt[m], self.box_kind).sum() / n
        else:
            box_loss = pred.sum() * 0.0
        total = self.w_hm * hm_loss + self.w_off * off_loss + self.w_box * box_loss
        return total, {"det_hm": hm_loss.item(), "det_off": off_loss.item(), "det_box": box_loss.item()}


# ----------------------------------------------------------------------------
# multi-task weighting
# ----------------------------------------------------------------------------
class TaskWeighting(nn.Module):
    """``uncertainty``: L = sum_i exp(-s_i) L_i + s_i  (Kendall et al., CVPR 2018),
    s_i = log sigma_i^2 learned jointly. ``fixed``: L = L_cls + w_det * L_det."""

    def __init__(self, mode: str, fixed_det_weight: float = 0.5):
        super().__init__()
        self.mode = mode
        self.fixed = fixed_det_weight
        self.log_vars = nn.Parameter(torch.zeros(2)) if mode == "uncertainty" else None

    def forward(self, l_cls, l_det):
        if self.mode == "uncertainty":
            s = self.log_vars
            return torch.exp(-s[0]) * l_cls + s[0] + torch.exp(-s[1]) * l_det + s[1]
        return l_cls + self.fixed * l_det

    def weights(self) -> dict:
        if self.mode == "uncertainty":
            w = torch.exp(-self.log_vars.detach()).cpu().tolist()
            return {"w_cls": w[0], "w_det": w[1]}
        return {"w_cls": 1.0, "w_det": self.fixed}
