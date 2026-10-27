"""Train + test one experiment on one fold.

    python scripts/train.py                                   # main model, fold 0
    python scripts/train.py --fold 3
    python scripts/train.py --name resnet50 --opts model.name=timm_classifier model.backbone=resnet50
    python scripts/train.py --resume                          # continue an interrupted run

Outputs go to results/runs/<name>/fold<k>/ (see README "Outputs").
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_path  # noqa: E402
from src.engine import Trainer  # noqa: E402
from src.utils import seed_everything  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--name", default=None, help="experiment name (overrides experiment.name)")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--opts", nargs="*", default=[], help="key.sub=value overrides")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    opts = list(args.opts)
    if args.name:
        opts.append(f"experiment.name={args.name}")
    if args.fold is not None:
        opts.append(f"data.fold={args.fold}")
    cfg = load_config(args.config, opts)
    seed_everything(cfg.experiment.seed + cfg.data.fold)

    run_dir = resolve_path(cfg.experiment.output_root) / cfg.experiment.name / f"fold{cfg.data.fold}"
    if (run_dir / "DONE").exists() and not args.resume:
        print(f"{run_dir} already finished (delete the DONE file to re-run)")
        return
    Trainer(cfg, run_dir, resume=args.resume).fit()


if __name__ == "__main__":
    main()
