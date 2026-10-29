"""Correctness tests for the parts a reviewer is most likely to question.

    python -m pytest tests -q        (or)        python tests/test_core.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_path  # noqa: E402
from src.metrics import (average_precision, bootstrap_ci, classification_metrics, holm_correction,  # noqa: E402
                         mcnemar_test)


def test_grouped_splits_are_leak_free():
    cfg = load_config()
    from src.data.splits import fold_frame

    full = pd.read_csv(resolve_path(cfg.data.splits_dir) / "grouped.csv.gz")
    tested = []
    for k in range(5):
        f = fold_frame(full, k, 5)
        per_group = f.groupby("group")["split"].nunique()
        assert (per_group == 1).all(), f"fold {k}: a near-duplicate group spans several splits"
        assert set(f.split) == {"train", "val", "test"}
        tested += f.loc[f.split == "test", "path"].tolist()
    assert len(tested) == len(set(tested)), "an image is tested in more than one fold"
    assert len(tested) == len(full), "not every image is tested exactly once"


def test_detection_roundtrip_is_exact():
    """Encoding GT boxes into CenterNet targets and decoding them back must give AP ~ 1.

    Not exactly 1: two adjacent stones whose centres fall into the same stride-4 cell
    collapse into one target (5 of 549 val+test boxes at 384 px, 1 at 512 px; AP ceiling
    0.990 at both sizes), an inherent limit of CenterNet."""
    import torch

    from src.data.datasets import build_det_loaders
    from src.engine import decode_centernet

    cfg = load_config(None, ["data.num_workers=0", "data.debug_max_samples=48", "train.det_batch_size=8"])
    preds, gts = [], []
    for b in build_det_loaders(cfg)["test"]:
        B, _, H, W = b["hm"].shape
        hm = b["hm"].clamp(1e-6, 1 - 1e-6)
        wh, off = torch.zeros(B, 2, H, W), torch.zeros(B, 2, H, W)
        for i in range(B):
            for k in range(int(b["num_boxes"][i])):
                ind = int(b["ind"][i, k])
                x1, y1, x2, y2 = b["boxes"][i, k]
                wh[i, :, ind // W, ind % W] = torch.tensor([(x2 - x1) / 4, (y2 - y1) / 4])
                off[i, :, ind // W, ind % W] = b["off"][i, k]
        preds += decode_centernet({"hm": torch.log(hm / (1 - hm)), "wh": wh, "off": off}, 4, 50)
        gts += [b["boxes"][i, :int(b["num_boxes"][i])].numpy() for i in range(B)]
    assert average_precision(preds, gts, 0.75) > 0.98


def test_ap_simple_cases():
    gt = [np.array([[0, 0, 10, 10]], np.float32)]
    assert average_precision([(np.array([[0, 0, 10, 10]], np.float32), np.array([0.9]))], gt, 0.5) == 1.0
    assert average_precision([(np.array([[50, 50, 60, 60]], np.float32), np.array([0.9]))], gt, 0.5) == 0.0


def test_classification_metrics_and_stats():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 400)
    probs = np.eye(4)[y] * 0.7 + 0.075
    m = classification_metrics(y, probs, ["Cyst", "Normal", "Stone", "Tumor"])
    assert abs(m["accuracy"] - 1.0) < 1e-9 and abs(m["macro_f1"] - 1.0) < 1e-9
    assert m["stone_vs_rest"]["sensitivity"] == 1.0
    lo, hi = bootstrap_ci(y, probs, lambda a, b: float((b.argmax(1) == a).mean()), iters=50)
    assert lo == hi == 1.0
    assert mcnemar_test(np.ones(10, bool), np.ones(10, bool))["p_value"] == 1.0
    assert mcnemar_test(np.ones(30, bool), np.zeros(30, bool))["p_value"] < 1e-6
    assert holm_correction([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
