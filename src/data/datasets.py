"""PyTorch datasets and loader builders."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ..augmentation import cls_transforms, det_transforms, offline_variant
from ..config import resolve_path
from ..preprocessing import build_input, read_gray, to_tensor
from ..utils import worker_init_fn
from .splits import detection_items, fold_frame


# ----------------------------------------------------------------------------
# Classification (Dataset 2)
# ----------------------------------------------------------------------------
class KidneyClsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: Path, transform, input_mode: str):
        self.df = df.reset_index(drop=True)
        self.root = Path(data_root)
        self.transform = transform
        self.input_mode = input_mode
        self.has_aug = "aug_id" in self.df.columns

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        gray = read_gray(self.root / row["path"])
        if self.has_aug and int(row["aug_id"]) > 0:
            gray = offline_variant(gray, int(row["aug_id"]), seed=i)
        gray = self.transform(image=gray)["image"]
        x = to_tensor(build_input(gray, self.input_mode))
        return {"image": x, "label": torch.tensor(int(row["label"]), dtype=torch.long), "index": i}


# ----------------------------------------------------------------------------
# Detection (Dataset 1) - CenterNet targets
# ----------------------------------------------------------------------------
def read_yolo_boxes(label_path: Path, w: int, h: int) -> np.ndarray:
    """YOLO txt (class cx cy bw bh, normalized; polygons tolerated) -> Nx4 xyxy pixels."""
    boxes = []
    if label_path.exists():
        for line in label_path.read_text().strip().splitlines():
            v = [float(t) for t in line.split()]
            if len(v) < 5:
                continue
            if len(v) == 5:
                _, cx, cy, bw, bh = v
                x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
            else:  # polygon: class x1 y1 x2 y2 ...
                xs, ys = v[1::2], v[2::2]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            x1, x2 = np.clip([x1 * w, x2 * w], 0, w)
            y1, y2 = np.clip([y1 * h, y2 * h], 0, h)
            if x2 - x1 >= 1 and y2 - y1 >= 1:
                boxes.append([x1, y1, x2, y2])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def gaussian_radius(h: float, w: float, min_overlap: float = 0.7) -> float:
    """CenterNet (Zhou et al., 2019) radius so a shifted box keeps IoU >= min_overlap."""
    a1, b1 = 1, h + w
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(b1 ** 2 - 4 * a1 * c1)) / 2
    a2, b2 = 4, 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    r2 = (b2 + math.sqrt(b2 ** 2 - 4 * a2 * c2)) / 2
    a3, b3 = 4 * min_overlap, -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    r3 = (b3 + math.sqrt(b3 ** 2 - 4 * a3 * c3)) / 2
    return min(r1, r2, r3)


def draw_gaussian(hm: np.ndarray, cx: int, cy: int, radius: int) -> None:
    d = 2 * radius + 1
    sigma = d / 6
    ys, xs = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    g = np.exp(-(xs * xs + ys * ys) / (2 * sigma * sigma))
    H, W = hm.shape
    l, r = min(cx, radius), min(W - cx, radius + 1)
    t, b = min(cy, radius), min(H - cy, radius + 1)
    patch = hm[cy - t:cy + b, cx - l:cx + r]
    gp = g[radius - t:radius + b, radius - l:radius + r]
    np.maximum(patch, gp, out=patch)


class StoneDetDataset(Dataset):
    MAX_OBJS = 32

    def __init__(self, items, transform, input_mode: str, img_size: int, stride: int):
        self.items = items
        self.transform = transform
        self.input_mode = input_mode
        self.size = img_size
        self.stride = stride

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        img_path, lbl_path = self.items[i]
        gray = read_gray(img_path)
        h, w = gray.shape
        boxes = read_yolo_boxes(lbl_path, w, h)
        out = self.transform(image=gray, bboxes=boxes, labels=[0] * len(boxes))
        gray = out["image"]
        boxes = np.array(out["bboxes"], dtype=np.float32).reshape(-1, 4)[: self.MAX_OBJS]
        x = to_tensor(build_input(gray, self.input_mode))

        fs = self.size // self.stride
        hm = np.zeros((fs, fs), dtype=np.float32)
        ind = np.zeros(self.MAX_OBJS, dtype=np.int64)
        mask = np.zeros(self.MAX_OBJS, dtype=np.float32)
        off = np.zeros((self.MAX_OBJS, 2), dtype=np.float32)
        gt = np.zeros((self.MAX_OBJS, 4), dtype=np.float32)
        for k, (x1, y1, x2, y2) in enumerate(boxes):
            bw, bh = (x2 - x1) / self.stride, (y2 - y1) / self.stride
            cx, cy = (x1 + x2) / 2 / self.stride, (y1 + y2) / 2 / self.stride
            ci, cj = min(int(cx), fs - 1), min(int(cy), fs - 1)
            r = max(0, int(gaussian_radius(math.ceil(bh), math.ceil(bw))))
            draw_gaussian(hm, ci, cj, r)
            ind[k] = cj * fs + ci
            mask[k] = 1.0
            off[k] = (cx - ci, cy - cj)
            gt[k] = (x1, y1, x2, y2)
        return {
            "image": x,
            "hm": torch.from_numpy(hm)[None],
            "ind": torch.from_numpy(ind),
            "mask": torch.from_numpy(mask),
            "off": torch.from_numpy(off),
            "boxes": torch.from_numpy(gt),   # xyxy in input pixels (padded)
            "num_boxes": len(boxes),
            "index": i,
        }


# ----------------------------------------------------------------------------
# builders
# ----------------------------------------------------------------------------
def load_split_frame(cfg) -> pd.DataFrame:
    path = resolve_path(cfg.data.splits_dir) / f"{cfg.data.split_protocol}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"Split file {path} not found. Run:  python scripts/make_splits.py")
    df = pd.read_csv(path)
    return fold_frame(df, int(cfg.data.fold), int(df["part"].nunique()))


def class_weights(labels: np.ndarray, num_classes: int, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    w = 1.0 / counts if mode == "inverse" else 1.0 / np.sqrt(counts)
    w = w / w.sum() * num_classes
    return torch.tensor(w, dtype=torch.float32)


def _loader(ds, batch_size, shuffle, cfg, sampler=None, drop_last=False):
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle and sampler is None, sampler=sampler,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory and torch.cuda.is_available(),
        drop_last=drop_last, worker_init_fn=worker_init_fn,
        persistent_workers=cfg.data.num_workers > 0 and shuffle,  # only train loaders stay alive
    )


def build_cls_loaders(cfg) -> dict:
    df = load_split_frame(cfg)
    root = resolve_path(cfg.data.root)
    mode = cfg.data.input_mode
    out = {"frame": df}
    for split in ("train", "val", "test"):
        sdf = df[df["split"] == split].reset_index(drop=True)
        if cfg.data.debug_max_samples:
            per_class = max(1, int(cfg.data.debug_max_samples) // len(cfg.data.classes))
            sdf = sdf.groupby("label", group_keys=False).head(per_class).reset_index(drop=True)
        train = split == "train"
        ds = KidneyClsDataset(sdf, root, cls_transforms(cfg, train), mode)
        sampler = None
        if train and cfg.train.balanced_sampler:
            w = 1.0 / np.bincount(sdf["label"].values)[sdf["label"].values]
            sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w), replacement=True)
        bs = cfg.train.batch_size if train else cfg.train.batch_size * 2
        out[split] = _loader(ds, bs, train, cfg, sampler=sampler, drop_last=train)
    return out


def build_det_loaders(cfg) -> dict:
    root = resolve_path(cfg.data.root)
    out = {}
    for split, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        items = detection_items(root, cfg.data.det_dir, folder)
        if cfg.data.debug_max_samples:
            items = items[: int(cfg.data.debug_max_samples)]
        train = split == "train"
        ds = StoneDetDataset(items, det_transforms(cfg, train), cfg.data.input_mode,
                             cfg.data.det_img_size, cfg.model.det_stride)
        bs = cfg.train.det_batch_size
        out[split] = _loader(ds, bs, train, cfg, drop_last=train)
    return out
