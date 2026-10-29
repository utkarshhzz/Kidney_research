"""Download + extract both Kaggle datasets and verify the layout the splits expect.

    python scripts/download_data.py                                   # into data/
    python scripts/download_data.py --cache-dir /content/drive/MyDrive/Kidney_research/datasets   # Colab

Kaggle credentials: env vars KAGGLE_USERNAME / KAGGLE_KEY, or ~/.kaggle/kaggle.json.
With --cache-dir the zips are kept there (e.g. Google Drive) and re-used in
later sessions, so each new Colab session only copies + unzips (no re-download).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_path  # noqa: E402

DATASETS = {
    "dataset1": ("safurahajiheidari/kidney-stone-images", "kidney-stone-images.zip"),
    "dataset2": ("nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone",
                 "ct-kidney-dataset-normal-cyst-tumor-and-stone.zip"),
}
IMG = {".jpg", ".jpeg", ".png", ".bmp"}


def fetch(slug: str, zip_name: str, dest: Path) -> Path:
    z = dest / zip_name
    if z.exists() and zipfile.is_zipfile(z):
        print(f"  using cached {z}")
        return z
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {slug} ...", flush=True)
    rc = subprocess.call(["kaggle", "datasets", "download", "-d", slug, "-p", str(dest)])
    if rc != 0 or not z.exists():
        sys.exit("Kaggle download failed. Check your credentials (KAGGLE_USERNAME/KAGGLE_KEY or "
                 "~/.kaggle/kaggle.json) and that you accepted the dataset's terms on kaggle.com.")
    return z


def find_dir(base: Path, ok) -> Path | None:
    for d in [base, *sorted(p for p in base.rglob("*") if p.is_dir())]:
        if ok(d):
            return d
    return None


def place_dataset1(tmp: Path, target: Path) -> None:
    src = find_dir(tmp, lambda d: all((d / s / "images").is_dir() for s in ("train", "valid", "test")))
    if src is None:
        sys.exit("Dataset 1: could not find train/valid/test/images in the archive")
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(src), str(target))


def place_dataset2(tmp: Path, final: Path, classes) -> None:
    """Move the folder holding the class sub-folders to exactly ``final`` (= data/<cls_dir>)."""
    src = find_dir(tmp, lambda d: all((d / c).is_dir() for c in classes))
    if src is None:
        sys.exit(f"Dataset 2: could not find class folders {classes} in the archive")
    if final.exists():
        shutil.rmtree(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(final))


def verify(cfg, root: Path) -> None:
    det = root / cfg.data.det_dir
    counts = {s: sum(1 for p in (det / s / "images").iterdir() if p.suffix.lower() in IMG)
              for s in ("train", "valid", "test")}
    print(f"Dataset 1 images: {counts}  (expected train 1054 / valid 123 / test 123)")
    split_file = resolve_path(cfg.data.splits_dir) / "grouped.csv.gz"
    paths = pd.read_csv(split_file)["path"]
    missing = [p for p in paths if not (root / p).exists()]
    print(f"Dataset 2: {len(paths) - len(missing)}/{len(paths)} images referenced by the splits found")
    if missing:
        print("  first missing:", missing[:3])
        sys.exit("Dataset 2 layout/file names differ from the committed splits. "
                 "If the Kaggle dataset changed, regenerate splits: python scripts/make_splits.py")
    print("Data OK.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=None, help="keep/re-use the zip files here (e.g. Google Drive)")
    ap.add_argument("--force", action="store_true", help="re-extract even if data already present")
    args = ap.parse_args()

    cfg = load_config()
    root = resolve_path(cfg.data.root)
    root.mkdir(parents=True, exist_ok=True)
    have1 = (root / cfg.data.det_dir / "train" / "images").is_dir()
    have2 = all((root / cfg.data.cls_dir / c).is_dir() for c in cfg.data.classes)
    if have1 and have2 and not args.force:
        print("Both datasets already extracted.")
        verify(cfg, root)
        return

    cache = Path(args.cache_dir) if args.cache_dir else root / "_zips"
    for name, (slug, zip_name) in DATASETS.items():
        print(f"[{name}]")
        z = fetch(slug, zip_name, cache)
        tmp = root / f"_extract_{name}"
        if tmp.exists():
            shutil.rmtree(tmp)
        print(f"  extracting {z.name} ...", flush=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(tmp)
        if name == "dataset1":
            place_dataset1(tmp, root / cfg.data.det_dir)
        else:
            place_dataset2(tmp, root / cfg.data.cls_dir, cfg.data.classes)
        shutil.rmtree(tmp, ignore_errors=True)
    verify(cfg, root)


if __name__ == "__main__":
    main()
