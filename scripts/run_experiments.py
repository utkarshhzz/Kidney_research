"""Run the full experiment matrix (resumable: finished runs are skipped).

    python scripts/run_experiments.py --dry-run              # print the plan
    python scripts/run_experiments.py --paper --max-folds 3  # 4-page paper set (~3.5 T4-hours)
    python scripts/run_experiments.py --tier1                # minimum publishable set first
    python scripts/run_experiments.py --tier1 --max-folds 3  # budget plan (3 folds per experiment)
    python scripts/run_experiments.py --groups main baselines
    python scripts/run_experiments.py --only kidneymtl_main --folds 0 1
    python scripts/run_experiments.py --hours 10             # stop starting new runs after 10 h (Colab)
    python scripts/run_experiments.py                        # everything

Each run is a separate process (a crash or OOM in one run does not kill the
queue). A run that fails with CUDA out-of-memory is retried once with half the
batch size. Status is appended to <output_root>/_status.csv.

Finished runs trained with an older ``experiment.protocol_version`` (different
training settings) are stale: their folder is renamed to ``fold<k>_stale_v<old>``
(nothing is deleted) and the run is trained again, so a table never mixes settings.

``--paper`` runs fold-major (fold 0 of every experiment, then fold 1, ...), so if
the time budget runs out every experiment still has the same number of folds.
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_path  # noqa: E402
from src.experiments import EXPERIMENTS, GROUP_ORDER, PAPER, TIER1, experiment_opts, folds_for  # noqa: E402


def _launch(name, fold, extra_opts, out_root):
    cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), "--fold", str(fold), "--resume",
           "--opts", *experiment_opts(name), *extra_opts]
    return subprocess.call(cmd, cwd=ROOT)


def _run_version(run_dir: Path) -> int | None:
    """protocol_version of a finished run (1 = before versioning), None if not finished."""
    cfg = run_dir / "config.yaml"
    if not (run_dir / "DONE").exists() or not cfg.exists():
        return None
    m = re.search(r"^\s*protocol_version:\s*(\d+)", cfg.read_text(errors="ignore"), re.M)
    return int(m.group(1)) if m else 1


def _archive(path: Path, version: int):
    """Rename a stale run folder (never delete results)."""
    if not path.exists():
        return
    dst, i = path.with_name(f"{path.name}_stale_v{version}"), 1
    while dst.exists():
        dst, i = path.with_name(f"{path.name}_stale_v{version}_{i}"), i + 1
    shutil.move(str(path), str(dst))
    print(f"--- archived stale run {path} -> {dst.name}")


def _oom(out_root: Path, name: str, fold: int) -> bool:
    log = out_root / name / f"fold{fold}" / "error.log"
    return log.exists() and "out of memory" in log.read_text(errors="ignore").lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="*", default=None, choices=GROUP_ORDER)
    ap.add_argument("--only", nargs="*", default=None, help="experiment names")
    ap.add_argument("--tier1", action="store_true", help="only the minimum publishable set")
    ap.add_argument("--paper", action="store_true", help="only the experiments of the 4-page paper (fold-major)")
    ap.add_argument("--folds", nargs="*", type=int, default=None, help="override folds for all selected")
    ap.add_argument("--max-folds", type=int, default=None, help="use at most the first N folds per experiment")
    ap.add_argument("--hours", type=float, default=None, help="do not start new runs after this many hours")
    ap.add_argument("--opts", nargs="*", default=[], help="extra overrides applied to every run")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    args = ap.parse_args()

    names = [n for g in GROUP_ORDER for n, e in EXPERIMENTS.items() if e["group"] == g]
    if args.tier1:
        names = list(TIER1)
    if args.paper:
        names = list(PAPER)
    if args.groups:
        names = [n for n in names if EXPERIMENTS[n]["group"] in args.groups]
    if args.only:
        unknown = set(args.only) - set(EXPERIMENTS)
        if unknown:
            sys.exit(f"unknown experiments: {unknown}")
        names = [n for n in names if n in args.only] or list(args.only)

    cfg = load_config(None, args.opts)
    out_root = resolve_path(cfg.experiment.output_root)
    current = int(cfg.experiment.get("protocol_version", 1))
    plan = []  # (name, fold, done, stale_version)
    for n in names:
        folds = args.folds if args.folds is not None else folds_for(n)
        if args.max_folds:
            folds = folds[: args.max_folds]
        for k in folds:
            v = _run_version(out_root / n / f"fold{k}")
            stale = v if v is not None and v != current else None
            plan.append((n, k, v is not None and stale is None, stale))
    if args.paper:
        plan.sort(key=lambda p: (p[1], names.index(p[0])))

    todo = [p for p in plan if not p[2]]
    print(f"{len(plan)} runs selected, {len(plan) - len(todo)} already done, {len(todo)} to run")
    for n, k, done, stale in plan:
        note = f"  STALE (settings v{stale}, now v{current}) -> archived and re-trained" if stale else ""
        print(f"  [{'x' if done else ' '}] {n:24s} fold {k}  ({EXPERIMENTS[n]['group']}){note}")
    if args.dry_run:
        return

    status = out_root / "_status.csv"
    status.parent.mkdir(parents=True, exist_ok=True)
    work_root = resolve_path(cfg.experiment.work_dir) if cfg.experiment.get("work_dir") else None
    t_start = time.time()
    for i, (n, k, _, stale) in enumerate(todo, 1):
        if args.hours and (time.time() - t_start) / 3600 > args.hours:
            print(f"\nTime budget of {args.hours} h reached - stopping. Re-run the same command to continue.")
            break
        if stale:
            _archive(out_root / n / f"fold{k}", stale)
            if work_root is not None:  # old last.pt must not be resumed
                _archive(work_root / n / f"fold{k}", stale)
        print(f"\n=== [{i}/{len(todo)}] {n} fold {k} ===", flush=True)
        t0 = time.time()
        rc = _launch(n, k, args.opts, out_root)
        if rc != 0 and _oom(out_root, n, k):
            run_cfg = load_config(None, experiment_opts(n) + list(args.opts))
            half = [f"train.batch_size={max(4, run_cfg.train.batch_size // 2)}",
                    f"train.det_batch_size={max(2, run_cfg.train.det_batch_size // 2)}"]
            print(f"--- CUDA out of memory: retrying {n} fold {k} with {half}", flush=True)
            rc = _launch(n, k, list(args.opts) + half, out_root)
        with open(status, "a", newline="") as f:
            csv.writer(f).writerow([datetime.now().isoformat(timespec="seconds"), n, k, rc,
                                    round((time.time() - t0) / 60, 1)])
        if rc != 0:
            print(f"!!! {n} fold {k} failed with exit code {rc} (traceback in its error.log)")
            if args.stop_on_error:
                sys.exit(rc)


if __name__ == "__main__":
    main()
