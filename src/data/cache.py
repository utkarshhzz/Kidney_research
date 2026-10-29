"""On-disk cache of resized classification slices (Colab speed-up).

Decoding a 569x705 JPEG and resizing it costs more CPU than the rest of the
per-sample pipeline, and Colab has only 2 vCPUs. The cache stores every slice
once as uint8 at ``data.cls_img_size`` in a single .npy file that is memory-mapped
by all loader workers (shared through the OS page cache). Augmentation and the
CLAHE/contour channels are still computed online, after augmentation, exactly as
without the cache, so results do not depend on whether the cache is used.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from ..preprocessing import read_gray


def _resize(path: Path, size: int) -> np.ndarray:
    return cv2.resize(read_gray(path), (size, size), interpolation=cv2.INTER_AREA)


def gray_cache(cache_dir, data_root, rel_paths, size: int, log=print) -> tuple[Path, dict]:
    """Return (.npy file with N x size x size uint8 slices, {relative path: row}) for ``rel_paths``.

    The dataset opens the file lazily with ``np.load(mmap_mode="r")`` in each loader
    worker, so the array is never pickled into spawned workers.

    Built on first use (about a minute on Colab) and reused by later runs in the
    same session. A path list that differs from the cached one rebuilds the file.
    """
    rel_paths = sorted(set(rel_paths))
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    arr_f, idx_f = cache_dir / f"cls_gray_{size}.npy", cache_dir / f"cls_gray_{size}.paths.txt"
    if not (arr_f.exists() and idx_f.exists() and idx_f.read_text().split("\n") == rel_paths):
        log(f"building slice cache {arr_f} ({len(rel_paths)} images at {size}px)")
        root = Path(data_root)
        tmp = arr_f.with_suffix(".tmp.npy")
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(rel_paths), size, size))
        with ThreadPoolExecutor(max_workers=max(2, os.cpu_count() or 2)) as ex:
            for i, img in enumerate(ex.map(lambda p: _resize(root / p, size), rel_paths, chunksize=64)):
                out[i] = img
        out.flush()
        del out
        os.replace(tmp, arr_f)
        idx_f.write_text("\n".join(rel_paths))
    return arr_f, {p: i for i, p in enumerate(rel_paths)}
