"""Input representations for CT slices.

The reference paper (CDR-VGG19, Devi et al. 2025) *replaces* the CT image with a
contour-detected image. We instead stack complementary views into the three
input channels so an ImageNet-pretrained backbone can use all of them:

    tri = [ grayscale , CLAHE(grayscale) , multi-level contour map ]

Every representation below is an ablation option (``data.input_mode``).
All functions take/return uint8 arrays so they can run after augmentation.
"""
from __future__ import annotations

import cv2
import numpy as np

# OpenCV's own thread pool inside DataLoader worker processes causes
# oversubscription and sporadic crashes (seen as segfaults on Windows).
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

INPUT_MODES = ("gray", "clahe", "contour", "sobel", "canny", "otsu", "tri")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_CLAHE = None


def read_gray(path: str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def clahe(gray: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    global _CLAHE
    if _CLAHE is None:  # cv2 CLAHE objects are not picklable -> build lazily per worker
        _CLAHE = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return _CLAHE.apply(gray)


def sobel(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def canny(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    v = float(np.median(g))
    lo, hi = int(max(0, 0.66 * v)), int(min(255, 1.33 * v))
    return cv2.Canny(g, lo, max(hi, lo + 1))


def otsu(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def contour_map(gray: np.ndarray, percentiles=(85.0, 95.0, 99.0), thickness: int = 1) -> np.ndarray:
    """Multi-level iso-intensity contour map.

    The reference paper's contour-detected filter (grayscale -> threshold ->
    contours -> boundary fill) with a single global threshold mostly outlines
    the body. Stones are hyper-dense, so we also trace contours at high
    intensity percentiles; brighter levels are drawn with brighter values so
    the map encodes *which* iso-level a boundary belongs to.
    """
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu_t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = g[g > otsu_t]
    levels = [otsu_t]
    if fg.size > 0:
        levels += [float(np.percentile(fg, p)) for p in percentiles]
    out = np.zeros_like(gray)
    n = len(levels)
    for i, t in enumerate(sorted(set(levels))):
        _, th = cv2.threshold(g, t, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        value = int(255 * (i + 1) / n)
        cv2.drawContours(out, contours, -1, value, thickness)
    return out


def build_input(gray: np.ndarray, mode: str = "tri") -> np.ndarray:
    """uint8 HxW -> uint8 HxWx3 according to ``mode``."""
    if mode == "gray":
        chans = [gray, gray, gray]
    elif mode == "clahe":
        c = clahe(gray)
        chans = [c, c, c]
    elif mode == "contour":
        k = contour_map(gray)
        chans = [k, k, k]
    elif mode == "sobel":
        s = sobel(gray)
        chans = [s, s, s]
    elif mode == "canny":
        e = canny(gray)
        chans = [e, e, e]
    elif mode == "otsu":
        o = otsu(gray)
        chans = [o, o, o]
    elif mode == "tri":
        chans = [gray, clahe(gray), contour_map(gray)]
    else:
        raise ValueError(f"Unknown input_mode '{mode}', choose from {INPUT_MODES}")
    return np.stack(chans, axis=-1)


def to_tensor(img3: np.ndarray):
    """uint8 HxWx3 -> normalized float32 CxHxW torch tensor."""
    import torch

    x = img3.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def denormalize(t) -> np.ndarray:
    """CxHxW normalized tensor -> uint8 HxWx3 (for figures)."""
    x = t.detach().cpu().numpy().transpose(1, 2, 0)
    x = x * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)
