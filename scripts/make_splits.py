"""Build all split files and the leakage audit.

    python scripts/make_splits.py                    # pHash grouping (default)
    python scripts/make_splits.py --method hybrid    # + CNN-embedding links (needs torch, slower)

Writes
    data/splits/{grouped,random,aug_then_split}.csv.gz  (path, label, group, part[, aug_id])
    data/splits/groups.csv.gz        (path, label, pHash, md5, group)
    data/splits/leakage_report.json  (numbers for the paper's leakage section)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_path  # noqa: E402
from src.data.splits import (assign_parts, compute_hashes, embedding_pairs, expand_offline,  # noqa: E402
                             fold_frame, groups_from_pairs, hamming_pairs, leakage_report,
                             list_cls_images)
from src.utils import save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--method", choices=["phash", "hybrid"], default="phash")
    ap.add_argument("--hash-threshold", type=int, default=4,
                    help="max Hamming distance (of 64 bits) to call two slices near-duplicates. "
                         "4 is the strictest value that still yields usable folds (6 merges >45%% of "
                         "the data into one group via transitive chaining)")
    ap.add_argument("--recompute-hashes", action="store_true")
    ap.add_argument("--embed-threshold", type=float, default=0.95)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--offline-variants", type=int, default=5,
                    help="variants per image for the reference (aug_then_split) protocol")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = resolve_path(cfg.data.root)
    out_dir = resolve_path(cfg.data.splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg.experiment.seed

    df = list_cls_images(root, cfg.data.cls_dir, cfg.data.classes)
    print(f"{len(df)} images: {df['class_name'].value_counts().to_dict()}")

    cache = out_dir / "groups.csv.gz"
    cached = pd.read_csv(cache) if cache.exists() and not args.recompute_hashes else None
    if cached is not None and list(cached["path"]) == list(df["path"]):
        print("re-using cached hashes from", cache)
        hashes = np.array([np.uint64(int(x, 16)) for x in cached["hash"]], dtype=np.uint64)
        md5 = cached["md5"].tolist()
    else:
        hashes = compute_hashes([root / p for p in df["path"]])
        md5 = [hashlib.md5((root / p).read_bytes()).hexdigest() for p in df["path"]]
    df["md5"] = md5
    dup_sets = df.groupby("md5").size()
    pairs = [(i, j) for i, j, _ in hamming_pairs(hashes, args.hash_threshold)]
    print(f"pHash near-duplicate pairs (<= {args.hash_threshold} bits): {len(pairs)}")
    if args.method == "hybrid":
        emb = list(embedding_pairs([root / p for p in df["path"]], args.embed_threshold))
        print(f"embedding pairs (cos >= {args.embed_threshold}): {len(emb)}")
        pairs += emb
    df["group"] = groups_from_pairs(len(df), pairs)
    df["hash"] = [format(int(h), "016x") for h in hashes]

    sizes = df.groupby("group").size()
    mixed = df.groupby("group")["label"].nunique()
    print(f"groups: {len(sizes)} | singletons: {(sizes == 1).sum()} | largest: {sizes.max()} "
          f"| groups spanning >1 class: {(mixed > 1).sum()}")
    df.to_csv(cache, index=False)

    report = {
        "n_images": len(df),
        "method": args.method,
        "hash_threshold": args.hash_threshold,
        "n_near_duplicate_pairs": len(pairs),
        "n_groups": int(len(sizes)),
        "n_singletons": int((sizes == 1).sum()),
        "largest_group": int(sizes.max()),
        "mean_group_size": float(sizes.mean()),
        "groups_spanning_multiple_classes": int((mixed > 1).sum()),
        "images_in_multi_image_groups": int(sizes[sizes > 1].sum()),
        "exact_duplicate_file_sets": int((dup_sets > 1).sum()),
        "images_in_exact_duplicate_sets": int(dup_sets[dup_sets > 1].sum()),
        "protocols": {},
    }

    # One file per protocol: 'part' = which of the n folds' test set the row belongs to.
    # The split of fold k is derived at load time (src.data.splits.fold_frame).
    keep = ["path", "label", "class_name", "group", "part"]
    for protocol in ("grouped", "random"):
        d = df.copy()
        d["part"] = assign_parts(d, args.folds, seed, grouped=(protocol == "grouped"))
        d[keep].to_csv(out_dir / f"{protocol}.csv.gz", index=False)
        fold_reports = []
        for k in range(args.folds):
            f = fold_frame(d, k, args.folds)
            fold_reports.append(leakage_report(f, hashes))
            counts = f.groupby(["split", "class_name"]).size().unstack(fill_value=0)
            print(f"  {protocol} fold{k} class counts:\n{counts}")
        report["protocols"][protocol] = {
            key: float(np.mean([r[key] for r in fold_reports])) for key in fold_reports[0]
        }
        print(protocol, report["protocols"][protocol])

    # Reference protocol: augment first, then split (variant-level, leaky by design)
    ex = expand_offline(df.drop(columns=["group"]), args.offline_variants)
    ex["part"] = assign_parts(ex, args.folds, seed, grouped=False)
    ex[keep + ["aug_id"]].to_csv(out_dir / "aug_then_split.csv.gz", index=False)
    f0 = fold_frame(ex, 0, args.folds)
    test_src = set(f0.loc[f0.split == "test", "path"])
    train_src = set(f0.loc[f0.split == "train", "path"])
    report["protocols"]["aug_then_split"] = {
        "n_rows": len(ex),
        "test_source_images_also_in_train": float(len(test_src & train_src) / max(len(test_src), 1)),
    }
    print("aug_then_split", report["protocols"]["aug_then_split"])

    save_json(report, out_dir / "leakage_report.json")
    print(f"Saved splits + report to {out_dir}")


if __name__ == "__main__":
    main()
