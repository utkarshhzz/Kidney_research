"""External (cross-dataset, cross-view) validation of the classifiers.

Every Dataset-1 image contains at least one annotated stone, but Dataset 1 is
*coronal* CT from a different source while the classifiers were trained on
*axial* Dataset-2 slices. We therefore report stone sensitivity on the
held-out Dataset-1 test split (never used for training by any model, including
the detection branch of KidneyMTL). Expect a large domain gap - report it
honestly; it is a limitation and a motivation for multi-source training.

Every training run already does this automatically (external_metrics.json) and
scripts/aggregate_results.py tabulates it; this script re-runs it from checkpoints.

    python scripts/external_validation.py                    # all runs with best.pt
    python scripts/external_validation.py --experiments kidneymtl_main b_resnet50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.config import load_config, resolve_path  # noqa: E402
from src.engine import external_validation, load_run_model  # noqa: E402
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
        model, _ = load_run_model(run, device)
        ext = external_validation(cfg, model, device, amp=device.type == "cuda")
        rows.append({"experiment": exp, "fold": cfg.data.fold, "protocol": cfg.data.split_protocol, **ext})
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
