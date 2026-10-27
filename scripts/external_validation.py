"""External (cross-dataset, cross-view) validation of the classifiers.

Every Dataset-1 image contains at least one annotated stone, but Dataset 1 is
*coronal* CT from a different source while the classifiers were trained on
*axial* Dataset-2 slices. We therefore report stone sensitivity on the
held-out Dataset-1 test split (never used for training by any model, including
the detection branch of KidneyMTL). Expect a large domain gap - report it
honestly; it is a limitation and a motivation for multi-source training.

    python scripts/external_validation.py                    # all runs with best.pt
    python scripts/external_validation.py --experiments kidneymtl_main b_resnet50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.augmentation import cls_transforms  # noqa: E402
from src.config import load_config, resolve_path  # noqa: E402
from src.data.datasets import KidneyClsDataset  # noqa: E402
from src.data.splits import detection_items  # noqa: E402
from src.engine import load_run_model, predict_cls  # noqa: E402
from src.metrics import mean_std  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="results/runs")
    ap.add_argument("--experiments", nargs="*", default=None)
    ap.add_argument("--out", default="results/tables")
    args = ap.parse_args()

    runs_root = resolve_path(args.runs_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for ck in sorted(runs_root.glob("*/fold*/best.pt")):
        run = ck.parent
        exp = run.parent.name
        if args.experiments and exp not in args.experiments:
            continue
        cfg = load_config(run / "config.yaml")
        if cfg.train.det_only:
            continue
        root = resolve_path(cfg.data.root)
        items = detection_items(root, cfg.data.det_dir, "test")
        stone = list(cfg.data.classes).index("Stone")
        df = pd.DataFrame({"path": [p.relative_to(root).as_posix() for p, _ in items], "label": stone})
        ds = KidneyClsDataset(df, root, cls_transforms(cfg, False), cfg.data.input_mode)
        loader = DataLoader(ds, batch_size=32, num_workers=cfg.data.num_workers)
        model, _ = load_run_model(run, device)
        probs, _, _ = predict_cls(model, loader, device, amp=device.type == "cuda", tta_hflip=cfg.eval.tta_hflip)
        pred = probs.argmax(1)
        top2 = np.argsort(-probs, 1)[:, :2]
        rows.append({
            "experiment": exp, "fold": cfg.data.fold, "protocol": cfg.data.split_protocol, "n": len(df),
            "stone_sensitivity": float((pred == stone).mean()),
            "stone_in_top2": float((top2 == stone).any(1).mean()),
            "mean_p_stone": float(probs[:, stone].mean()),
            **{f"pred_frac_{c}": float((pred == i).mean()) for i, c in enumerate(cfg.data.classes)},
        })
        print(rows[-1])
    if not rows:
        print("No runs with best.pt found.")
        return
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(out / "external_validation_per_fold.csv", index=False)
    agg = []
    for exp, g in per_fold.groupby("experiment"):
        r = {"experiment": exp, "folds": len(g)}
        for c in ("stone_sensitivity", "stone_in_top2", "mean_p_stone"):
            m, s = mean_std(g[c])
            r[c] = f"{100 * m:.2f} ± {100 * s:.2f}"
        agg.append(r)
    pd.DataFrame(agg).to_csv(out / "external_validation.csv", index=False)
    print(pd.DataFrame(agg).to_string(index=False))


if __name__ == "__main__":
    main()
