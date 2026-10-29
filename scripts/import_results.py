"""Merge result zips downloaded from Colab into this project's results/ folder.

    python scripts/import_results.py results_20261101_2130.zip [more.zip ...]
    python scripts/aggregate_results.py      # then rebuild all tables locally (no GPU / torch needed)

Several zips (from different Colab sessions) can be imported one after another;
files with the same path are overwritten by the newer zip.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zips", nargs="+")
    args = ap.parse_args()
    for z in args.zips:
        with zipfile.ZipFile(z) as zf:
            members = zf.namelist()
            bad = [m for m in members if not m.startswith("results/") or ".." in Path(m).parts
                   or Path(m).is_absolute()]
            if bad:
                raise SystemExit(f"{z}: unexpected paths in archive (not produced by package_results.py): {bad[:3]}")
            zf.extractall(ROOT)
            print(f"{z}: {len(members)} files -> {ROOT / 'results'}")
    runs = [r for r in sorted((ROOT / "results" / "runs").glob("*/fold*/DONE")) if r.parent.name[4:].isdigit()]
    exps = sorted({r.parent.parent.name for r in runs})
    print(f"\n{len(runs)} finished runs across {len(exps)} experiments:")
    for e in exps:
        folds = sorted(int(r.parent.name[4:]) for r in runs if r.parent.parent.name == e)
        print(f"  {e:26s} folds {folds}")
    print("\nNext: python scripts/aggregate_results.py")


if __name__ == "__main__":
    main()
