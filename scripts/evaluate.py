"""Re-evaluate a finished run (e.g. with/without TTA) from its best.pt.

    python scripts/evaluate.py --run results/runs/kidneymtl_main/fold0
    python scripts/evaluate.py --run results/runs/kidneymtl_main/fold0 --no-tta --out results/eval_no_tta
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.config import load_config  # noqa: E402
from src.data.datasets import build_cls_loaders, build_det_loaders  # noqa: E402
from src.engine import evaluate_run, load_run_model  # noqa: E402
from src.models import has_detection  # noqa: E402
from src.utils import save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="results/runs/<exp>/fold<k>")
    ap.add_argument("--out", default=None, help="output dir (default: overwrite the run dir)")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--opts", nargs="*", default=[])
    args = ap.parse_args()

    run = Path(args.run)
    opts = list(args.opts) + (["eval.tta_hflip=false"] if args.no_tta else [])
    cfg = load_config(run / "config.yaml", opts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, epoch = load_run_model(run, device)
    out = Path(args.out) if args.out else run
    out.mkdir(parents=True, exist_ok=True)
    det = build_det_loaders(cfg) if has_detection(model) else None
    res = evaluate_run(cfg, model, build_cls_loaders(cfg), det, out, device,
                       amp=bool(cfg.train.amp) and device.type == "cuda", best_epoch=epoch)
    save_json(res["summary"], out / "summary.json")
    print({k: round(v, 4) for k, v in res["summary"].items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
