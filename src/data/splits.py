"""Leakage-aware train/val/test splits for the CT-KIDNEY dataset.

The dataset ships without patient IDs and contains many consecutive slices of
the same patient, which are near-identical. A random image-level split then
puts near-copies of test images in the training set.

We approximate patient-level splitting by
  1. hashing every image with a 64-bit DCT perceptual hash (pHash),
  2. linking images whose hashes differ by <= ``threshold`` bits
     (optionally also linking images with cosine-similar CNN embeddings),
  3. taking connected components (union-find) as groups, and
  4. running StratifiedGroupKFold so a group never straddles two folds.

Fold scheme (5 folds): fold k -> test = part k, val = part (k+1) % 5,
train = remaining 3 parts (60/20/20). Every image is tested exactly once,
so out-of-fold predictions cover the whole dataset.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
PROTOCOLS = ("grouped", "random", "aug_then_split")


# ----------------------------------------------------------------------------
# listing
# ----------------------------------------------------------------------------
def list_cls_images(data_root: Path, cls_dir: str, classes: list[str]) -> pd.DataFrame:
    rows = []
    base = Path(data_root) / cls_dir
    for label, cname in enumerate(classes):
        cdir = base / cname
        if not cdir.is_dir():
            raise FileNotFoundError(f"Class folder not found: {cdir}")
        for p in sorted(cdir.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                rows.append({"path": p.relative_to(data_root).as_posix(), "label": label, "class_name": cname})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# perceptual hashing
# ----------------------------------------------------------------------------
def phash(gray: np.ndarray, hash_size: int = 8, highfreq: int = 4) -> np.uint64:
    size = hash_size * highfreq
    img = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(img)[:hash_size, :hash_size]
    med = np.median(dct.flatten()[1:])  # ignore DC term
    bits = (dct > med).flatten()
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


def compute_hashes(paths: list[Path]) -> np.ndarray:
    from tqdm import tqdm

    out = np.zeros(len(paths), dtype=np.uint64)
    for i, p in enumerate(tqdm(paths, desc="pHash", ncols=80)):
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        out[i] = phash(g)
    return out


_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount64(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):  # numpy >= 2.0
        return np.bitwise_count(x)
    return _POP8[x.view(np.uint8)].reshape(*x.shape, 8).sum(-1)


def hamming_pairs(hashes: np.ndarray, threshold: int, chunk: int = 512):
    """Yield (i, j, d) for i<j with Hamming distance <= threshold."""
    n = len(hashes)
    for s in range(0, n, chunk):
        block = hashes[s:s + chunk, None] ^ hashes[None, :]
        d = _popcount64(block)
        ii, jj = np.nonzero(d <= threshold)
        ii = ii + s
        keep = ii < jj
        for i, j in zip(ii[keep], jj[keep]):
            yield int(i), int(j), int(d[i - s, j])


def nearest_hamming(query: np.ndarray, ref: np.ndarray, chunk: int = 512) -> np.ndarray:
    """For each query hash, Hamming distance to the closest reference hash."""
    out = np.empty(len(query), dtype=np.int32)
    for s in range(0, len(query), chunk):
        d = _popcount64(query[s:s + chunk, None] ^ ref[None, :])
        out[s:s + chunk] = d.min(1)
    return out


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def groups_from_pairs(n: int, pairs) -> np.ndarray:
    uf = UnionFind(n)
    for i, j, *_ in pairs:
        uf.union(i, j)
    roots = np.array([uf.find(i) for i in range(n)])
    _, groups = np.unique(roots, return_inverse=True)
    return groups


def embedding_pairs(paths: list[Path], sim_threshold: float = 0.95, batch: int = 64):
    """Optional: link images whose ImageNet-ResNet50 embeddings are near-identical."""
    import timm
    import torch
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("resnet50", pretrained=True, num_classes=0).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    feats = []
    with torch.no_grad():
        for s in tqdm(range(0, len(paths), batch), desc="embed", ncols=80):
            ims = [cv2.resize(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), (224, 224)) for p in paths[s:s + batch]]
            x = torch.from_numpy(np.stack(ims)).float().div(255).unsqueeze(1).repeat(1, 3, 1, 1).to(device)
            f = model((x - mean) / std)
            feats.append(torch.nn.functional.normalize(f, dim=1).cpu())
    feats = torch.cat(feats)
    for s in range(0, len(feats), 1024):
        sim = feats[s:s + 1024] @ feats.T
        ii, jj = torch.nonzero(sim >= sim_threshold, as_tuple=True)
        ii = ii + s
        keep = ii < jj
        for i, j in zip(ii[keep].tolist(), jj[keep].tolist()):
            yield i, j


# ----------------------------------------------------------------------------
# fold construction
# ----------------------------------------------------------------------------
def assign_parts(df: pd.DataFrame, n_folds: int, seed: int, grouped: bool) -> np.ndarray:
    parts = np.full(len(df), -1)
    if grouped:
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        it = splitter.split(df, df["label"], groups=df["group"])
    else:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        it = splitter.split(df, df["label"])
    for k, (_, test_idx) in enumerate(it):
        parts[test_idx] = k
    assert (parts >= 0).all()
    return parts


def fold_frame(df: pd.DataFrame, fold: int, n_folds: int) -> pd.DataFrame:
    test_part, val_part = fold, (fold + 1) % n_folds
    out = df.copy()
    out["split"] = "train"
    out.loc[out["part"] == val_part, "split"] = "val"
    out.loc[out["part"] == test_part, "split"] = "test"
    return out


def expand_offline(df: pd.DataFrame, n_variants: int) -> pd.DataFrame:
    """Reference-paper protocol: every image becomes n_variants rows (aug_id)."""
    reps = [df.assign(aug_id=a) for a in range(n_variants)]
    out = pd.concat(reps, ignore_index=True)
    # each (image, variant) is its own 'group' -> variants scatter over splits
    out["group"] = np.arange(len(out))
    return out


def leakage_report(df: pd.DataFrame, hashes: np.ndarray, thresholds=(4, 8, 10, 12)) -> dict:
    """Fraction of test images whose nearest training image is a near-duplicate."""
    rep = {}
    test = df["split"].values == "test"
    train = df["split"].values == "train"
    dmin = nearest_hamming(hashes[test], hashes[train])
    for t in thresholds:
        rep[f"test_with_train_neighbour_le_{t}bits"] = float((dmin <= t).mean())
    rep["median_nn_distance"] = float(np.median(dmin))
    return rep


def detection_items(data_root: Path, det_dir: str, split: str) -> list[tuple[Path, Path]]:
    """Dataset 1 uses the Roboflow train/valid/test split (no source image
    crosses splits - verified by stripping the '.rf.<hash>' suffix)."""
    base = Path(data_root) / det_dir / split
    items = []
    for p in sorted((base / "images").iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            items.append((p, base / "labels" / (p.stem + ".txt")))
    return items
