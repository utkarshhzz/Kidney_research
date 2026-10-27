"""End-to-end smoke test (a few minutes on CPU). Run this first on any new machine.

    python scripts/smoke_test.py

Trains every model family for a handful of steps on tiny subsets (random
init, so no downloads), then runs evaluation, aggregation, explainability,
external validation and the efficiency benchmark on those runs. Nothing is
written to results/: outputs go to results/_smoke/ and are deleted at the end
(use --keep to inspect them).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments import experiment_opts  # noqa: E402

TINY = ["model.pretrained=false", "data.cls_img_size=128", "data.det_img_size=256", "data.num_workers=0",
        "data.debug_max_samples=48", "train.batch_size=8", "train.det_batch_size=4", "train.epochs=2",
        "train.max_steps_per_epoch=3", "train.warmup_epochs=1"]

CASES = {
    "kidneymtl_main": ["model.backbone=efficientnet_b0"],
    "a_no_det": ["model.backbone=efficientnet_b0"],
    "a_mtl_fixed": ["model.backbone=resnet18"],
    "b_resnet50": ["model.backbone=resnet18"],
    "b_vit_small": ["model.backbone=vit_tiny_patch16_224"],
    "b_cdr_vgg19": ["data.cls_img_size=224"],
    "d_det_only": ["model.backbone=efficientnet_b0"],
    "l_augsplit_kidneymtl": ["model.backbone=efficientnet_b0"],
}


def run(cmd):
    print("\n$", " ".join(cmd[1:]), flush=True)
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc != 0:
        sys.exit(f"FAILED (exit {rc}): {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--quick", action="store_true", help="only the main model")
    args = ap.parse_args()

    smoke = ROOT / "results" / "_smoke"
    runs = smoke / "runs"
    if smoke.exists():
        shutil.rmtree(smoke)
    py = sys.executable
    cases = {"kidneymtl_main": CASES["kidneymtl_main"]} if args.quick else CASES
    for name, extra in cases.items():
        run([py, "scripts/train.py", "--fold", "0", "--opts", *experiment_opts(name), *TINY, *extra,
             f"experiment.output_root={runs.as_posix()}", "experiment.keep_checkpoint=true"])
    run([py, "scripts/evaluate.py", "--run", str(runs / "kidneymtl_main" / "fold0"), "--no-tta",
         "--out", str(smoke / "eval_no_tta"), "--opts", "data.num_workers=0"])
    run([py, "scripts/aggregate_results.py", "--runs-root", str(runs), "--out", str(smoke), "--bootstrap", "50"])
    if not args.quick:
        run([py, "scripts/explain.py", "--runs-root", str(runs), "--experiments", "kidneymtl_main", "b_resnet50",
             "b_vit_small", "--per-class", "1", "--out", str(smoke / "explain")])
        run([py, "scripts/external_validation.py", "--runs-root", str(runs), "--out", str(smoke / "tables"),
             "--experiments", "kidneymtl_main", "b_resnet50"])
        run([py, "scripts/benchmark_efficiency.py", "--device", "cpu", "--iters", "2", "--warmup", "1",
             "--experiments", "kidneymtl_main", "b_resnet50", "--out", str(smoke / "tables")])
    print("\nSMOKE TEST PASSED - the pipeline runs end to end on this machine.")
    if not args.keep:
        shutil.rmtree(smoke)


if __name__ == "__main__":
    main()
