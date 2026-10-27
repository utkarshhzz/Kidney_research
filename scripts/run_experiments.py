"""Run the full experiment matrix (resumable: finished runs are skipped).

    python scripts/run_experiments.py --dry-run              # print the plan
    python scripts/run_experiments.py --tier1                # minimum publishable set first
    python scripts/run_experiments.py --groups main baselines
    python scripts/run_experiments.py --only kidneymtl_main --folds 0 1
    python scripts/run_experiments.py                        # everything

Each run is a separate process (a crash or OOM in one run does not kill the
queue). Status is appended to results/runs/_status.csv.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_path  # noqa: E402
from src.experiments import EXPERIMENTS, GROUP_ORDER, TIER1, experiment_opts, folds_for  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="*", default=None, choices=GROUP_ORDER)
    ap.add_argument("--only", nargs="*", default=None, help="experiment names")
    ap.add_argument("--tier1", action="store_true", help="only the minimum publishable set")
    ap.add_argument("--folds", nargs="*", type=int, default=None, help="override folds for all selected")
    ap.add_argument("--opts", nargs="*", default=[], help="extra overrides applied to every run")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    args = ap.parse_args()

    names = [n for g in GROUP_ORDER for n, e in EXPERIMENTS.items() if e["group"] == g]
    if args.tier1:
        names = [n for n in TIER1]
    if args.groups:
        names = [n for n in names if EXPERIMENTS[n]["group"] in args.groups]
    if args.only:
        unknown = set(args.only) - set(EXPERIMENTS)
        if unknown:
            sys.exit(f"unknown experiments: {unknown}")
        names = [n for n in names if n in args.only] or list(args.only)

    out_root = resolve_path(load_config(None, args.opts).experiment.output_root)
    plan = []
    for n in names:
        for k in (args.folds if args.folds is not None else folds_for(n)):
            done = (out_root / n / f"fold{k}" / "DONE").exists()
            plan.append((n, k, done))

    todo = [p for p in plan if not p[2]]
    print(f"{len(plan)} runs selected, {len(plan) - len(todo)} already done, {len(todo)} to run")
    for n, k, done in plan:
        print(f"  [{'x' if done else ' '}] {n:24s} fold {k}  ({EXPERIMENTS[n]['group']})")
    if args.dry_run:
        return

    status = out_root / "_status.csv"
    status.parent.mkdir(parents=True, exist_ok=True)
    for i, (n, k, _) in enumerate(todo, 1):
        cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), "--fold", str(k),
               "--opts", *experiment_opts(n), *args.opts]
        resume = (out_root / n / f"fold{k}" / "last.pt").exists()
        if resume:
            cmd.append("--resume")
        print(f"\n=== [{i}/{len(todo)}] {n} fold {k} {'(resuming)' if resume else ''} ===", flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=ROOT)
        with open(status, "a", newline="") as f:
            csv.writer(f).writerow([datetime.now().isoformat(timespec="seconds"), n, k, rc,
                                    round((time.time() - t0) / 60, 1)])
        if rc != 0:
            print(f"!!! {n} fold {k} failed with exit code {rc}")
            if args.stop_on_error:
                sys.exit(rc)


if __name__ == "__main__":
    main()
