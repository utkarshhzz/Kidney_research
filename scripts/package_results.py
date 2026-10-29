"""Zip all results (metrics, predictions, logs, plots, tables, figures) for download.

    python scripts/package_results.py                                        # results/ -> results_<time>.zip
    python scripts/package_results.py --results-dir /content/drive/MyDrive/Kidney_research/results \
                                      --out /content/drive/MyDrive/Kidney_research/downloads

Model checkpoints (*.pt) are skipped unless --include-checkpoints, so the zip
stays small (typically a few hundred MB at most). Inside the zip every file is
stored under results/..., so on the local machine
    python scripts/import_results.py results_<time>.zip
puts it in the right place no matter where it lived on Colab/Drive.
"""
from __future__ import annotations

import argparse
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"_smoke", "__pycache__"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT), help="directory for the zip")
    ap.add_argument("--include-checkpoints", action="store_true")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    src = Path(args.results_dir)
    if not src.is_dir():
        raise SystemExit(f"{src} does not exist")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"results_{datetime.now().strftime('%Y%m%d_%H%M')}.zip"
    zpath = out_dir / name

    n_files, n_runs, total = 0, 0, 0
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(src.rglob("*")):
            parts = p.relative_to(src).parts
            if not p.is_file() or any(part in SKIP_DIRS or "_stale_v" in part for part in parts):
                continue  # archived runs trained with older settings stay on Drive only
            if p.suffix in (".pt", ".pth") and not args.include_checkpoints:
                continue
            if p.resolve() == zpath.resolve() or p.suffix == ".zip":
                continue
            zf.write(p, ("results" / p.relative_to(src)).as_posix())
            n_files += 1
            total += p.stat().st_size
            n_runs += p.name == "DONE"
    print(f"{n_runs} finished runs, {n_files} files ({total / 1e6:.1f} MB raw) -> {zpath} "
          f"({zpath.stat().st_size / 1e6:.1f} MB)")
    print(zpath)


if __name__ == "__main__":
    main()
