"""Data augmentation.

IMPORTANT (methodology): augmentation is applied *online* and *only to the
training split*. The earlier notebook (03_data_augmentation.ipynb) and the
reference paper augmented every image 21x **before** splitting, so near-copies
of a test image are seen during training and test accuracy is inflated. We keep
that offline scheme here (``offline_variant``) only to reproduce the reference
protocol in the leakage study (``data.split_protocol=aug_then_split``).
"""
from __future__ import annotations

import cv2
import numpy as np
import albumentations as A


# ----------------------------------------------------------------------------
# Online augmentation (albumentations 2.x API)
# ----------------------------------------------------------------------------
def _geometric(cfg, size: int):
    a = cfg.augment
    return [
        A.HorizontalFlip(p=a.hflip),
        A.VerticalFlip(p=a.vflip),
        A.Affine(
            scale=(1 - a.scale_limit, 1 + a.scale_limit),
            translate_percent=(-a.shift_limit, a.shift_limit),
            rotate=(-a.rotate_limit, a.rotate_limit),
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=0.7,
        ),
    ]


def _photometric(cfg):
    a = cfg.augment
    return [
        A.RandomBrightnessContrast(brightness_limit=a.brightness_contrast,
                                   contrast_limit=a.brightness_contrast, p=0.5),
        A.RandomGamma(gamma_limit=(int(100 * (1 - a.gamma)), int(100 * (1 + a.gamma))), p=0.3),
        A.GaussNoise(std_range=(0.01, 0.05), p=a.gauss_noise),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.MotionBlur(blur_limit=(3, 5))], p=a.blur),
    ]


def cls_transforms(cfg, train: bool) -> A.Compose:
    size = cfg.data.cls_img_size
    if not train:
        return A.Compose([A.Resize(size, size)])
    a = cfg.augment
    return A.Compose(
        [A.Resize(size, size)]
        + _geometric(cfg, size)
        + _photometric(cfg)
        + [A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.05, 0.12),
                           hole_width_range=(0.05, 0.12), fill=0, p=a.coarse_dropout)]
    )


def det_transforms(cfg, train: bool) -> A.Compose:
    """Letterbox to a square canvas; boxes in absolute pascal_voc coords."""
    size = cfg.data.det_img_size
    base = [
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT,
                      fill=0, position="center"),
    ]
    aug = (_geometric(cfg, size) + _photometric(cfg)) if train else []
    return A.Compose(
        base + aug,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"],
                                 min_visibility=0.3, min_area=4),
    )


# ----------------------------------------------------------------------------
# Offline variants (reference-paper protocol replication ONLY)
# Mirrors the 21 hand-written augmentations from the earlier notebook.
# ----------------------------------------------------------------------------
def _rotate(img, angle):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def _translate(img, tx, ty):
    h, w = img.shape[:2]
    m = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)


def _zoom(img, f):
    h, w = img.shape[:2]
    if f >= 1:
        nh, nw = int(h / f), int(w / f)
        y, x = (h - nh) // 2, (w - nw) // 2
        return cv2.resize(img[y:y + nh, x:x + nw], (w, h))
    small = cv2.resize(img, (int(w * f), int(h * f)))
    out = np.zeros_like(img)
    y, x = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    out[y:y + small.shape[0], x:x + small.shape[1]] = small
    return out


def _contrast(img, f):
    m = img.mean()
    return np.clip((img.astype(np.float32) - m) * f + m, 0, 255).astype(np.uint8)


def _brightness(img, f):
    return np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)


def _noise(img, seed):
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.float32) + rng.normal(0, 25, img.shape), 0, 255).astype(np.uint8)


def _elastic(img, seed, alpha=50, sigma=5):
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    dx = cv2.GaussianBlur((rng.random((h, w)) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((rng.random((h, w)) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    return cv2.remap(img, (x + dx).astype(np.float32), (y + dy).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


_OFFLINE = [
    lambda g, s: g,                                   # 0 original
    lambda g, s: _rotate(g, 90),
    lambda g, s: _rotate(g, 180),
    lambda g, s: _rotate(g, 270),
    lambda g, s: _rotate(g, 45),
    lambda g, s: cv2.flip(g, 1),
    lambda g, s: cv2.flip(g, 0),
    lambda g, s: _brightness(g, 1.3),
    lambda g, s: _brightness(g, 0.7),
    lambda g, s: _contrast(g, 1.3),
    lambda g, s: _contrast(g, 0.7),
    lambda g, s: _noise(g, s),
    lambda g, s: cv2.GaussianBlur(g, (5, 5), 0),
    lambda g, s: cv2.filter2D(g, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])),
    lambda g, s: _zoom(g, 1.2),
    lambda g, s: _zoom(g, 0.8),
    lambda g, s: _translate(g, 20, 0),
    lambda g, s: _translate(g, -20, 0),
    lambda g, s: _translate(g, 0, 20),
    lambda g, s: _translate(g, 0, -20),
    lambda g, s: _elastic(g, s),
    lambda g, s: cv2.createCLAHE(2.0, (8, 8)).apply(g),
]
NUM_OFFLINE_VARIANTS = len(_OFFLINE)


def offline_variant(gray: np.ndarray, aug_id: int, seed: int = 0) -> np.ndarray:
    """Deterministic variant ``aug_id`` of an image (0 = original)."""
    return _OFFLINE[aug_id % NUM_OFFLINE_VARIANTS](gray, seed + aug_id)
