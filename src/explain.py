"""Grad-CAM (Selvaraju et al., ICCV 2017) and Grad-CAM++ (Chattopadhay et al., WACV 2018).

Every model stores its last spatial feature map in ``model.cam_features``
during the forward pass, so one implementation serves all architectures.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def class_activation_maps(model, x: torch.Tensor, target: torch.Tensor | int | None = None,
                          method: str = "gradcam"):
    """Returns (cams in [0,1] of shape (B,H,W) at input resolution, probs (B,K))."""
    model.eval()
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        x = x.clone().requires_grad_(True)  # guarantees a graph even if params are frozen
        logits = model(x, "cls")["logits"].float()
        feats = model.cam_features
        feats.retain_grad()
        if target is None:
            target = logits.argmax(1)
        elif isinstance(target, int):
            target = torch.full((len(x),), target, device=x.device, dtype=torch.long)
        logits.gather(1, target[:, None]).sum().backward()
    to_spatial = getattr(model, "to_spatial", lambda t: t)
    A = to_spatial(feats.detach()).float()
    G = to_spatial(feats.grad.detach()).float()
    if method == "gradcam":
        w = G.mean((2, 3), keepdim=True)
    elif method == "gradcam++":
        g2, g3 = G.pow(2), G.pow(3)
        denom = 2 * g2 + A.sum((2, 3), keepdim=True) * g3
        alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
        w = (alpha * F.relu(G)).sum((2, 3), keepdim=True)
    else:
        raise ValueError(method)
    cam = F.relu((w * A).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    flat = cam.view(len(cam), -1)
    lo, hi = flat.min(1, keepdim=True)[0], flat.max(1, keepdim=True)[0]
    cam = ((flat - lo) / (hi - lo).clamp(min=1e-8)).view_as(cam)
    model.zero_grad(set_to_none=True)
    return cam.cpu().numpy(), F.softmax(logits.detach(), 1).cpu().numpy()


def pointing_game_hit(cam: np.ndarray, boxes: np.ndarray, tolerance: int = 8) -> bool:
    """Zhang et al. (ECCV 2016): is the CAM maximum inside (or within ``tolerance`` px of) a GT box?"""
    y, x = np.unravel_index(int(cam.argmax()), cam.shape)
    for x1, y1, x2, y2 in boxes:
        if x1 - tolerance <= x <= x2 + tolerance and y1 - tolerance <= y <= y2 + tolerance:
            return True
    return False


def energy_in_boxes(cam: np.ndarray, boxes: np.ndarray) -> float:
    """Fraction of CAM mass falling inside GT boxes (energy-based pointing game, Score-CAM paper)."""
    mask = np.zeros_like(cam, dtype=bool)
    for x1, y1, x2, y2 in boxes.astype(int):
        mask[max(y1, 0):y2 + 1, max(x1, 0):x2 + 1] = True
    total = cam.sum()
    return float(cam[mask].sum() / total) if total > 0 else 0.0


def overlay(gray_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import cv2

    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    return np.clip((1 - alpha) * gray_rgb + alpha * heat, 0, 255).astype(np.uint8)
