"""Explainability: qualitative Grad-CAM figures + quantitative localisation.

1. Figure: test images of each class (Dataset 2) with Grad-CAM / Grad-CAM++.
2. Figure: same images across models (ours vs baselines).
3. Pointing game on the Dataset-1 test split (stone boxes available): is the
   Stone-class CAM peak inside an annotated stone? + CAM energy inside boxes.
   For KidneyMTL we also report the same score for the detection heatmap.

    python scripts/explain.py                                    # main model, fold 0
    python scripts/explain.py --experiments kidneymtl_main b_resnet50 b_cdr_vgg19 --fold 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.augmentation import det_transforms  # noqa: E402
from src.config import load_config, resolve_path  # noqa: E402
from src.data.datasets import read_yolo_boxes  # noqa: E402
from src.data.splits import detection_items  # noqa: E402
from src.engine import decode_centernet, load_run_model  # noqa: E402
from src.explain import class_activation_maps, energy_in_boxes, overlay, pointing_game_hit  # noqa: E402
from src.models import has_detection  # noqa: E402
from src.preprocessing import build_input, read_gray, to_tensor  # noqa: E402
from src.utils import save_json  # noqa: E402


def prep(gray, size, mode):
    g = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return g, to_tensor(build_input(g, mode))


def pick_examples(run: Path, classes, per_class: int, seed: int = 0) -> pd.DataFrame:
    df = pd.read_csv(run / "test_predictions.csv")
    out = []
    for i, _ in enumerate(classes):
        sub = df[(df.label == i) & (df.pred == i)]
        if len(sub):
            out.append(sub.sample(min(per_class, len(sub)), random_state=seed))
    return pd.concat(out).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="results/runs")
    ap.add_argument("--experiments", nargs="*", default=["kidneymtl_main"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--out", default="results/explain")
    args = ap.parse_args()

    runs_root = resolve_path(args.runs_root)
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = [runs_root / e / f"fold{args.fold}" for e in args.experiments]
    runs = [r for r in runs if (r / "best.pt").exists()]
    if not runs:
        sys.exit("no runs with best.pt found")

    ref_cfg = load_config(runs[0] / "config.yaml")
    classes = list(ref_cfg.data.classes)
    root = resolve_path(ref_cfg.data.root)
    examples = pick_examples(runs[0], classes, args.per_class)
    stone = classes.index("Stone")
    det_items = detection_items(root, ref_cfg.data.det_dir, "test")
    scores, comparison = {}, {}

    for run in runs:
        exp = run.parent.name
        cfg = load_config(run / "config.yaml")
        model, _ = load_run_model(run, device)
        size, mode = cfg.data.cls_img_size, cfg.data.input_mode

        # ---- qualitative figure (per model)
        fig, axes = plt.subplots(len(examples), 3, figsize=(7.5, 2.5 * len(examples)))
        axes = np.atleast_2d(axes)
        comparison[exp] = []
        for r, row in examples.iterrows():
            g, x = prep(read_gray(root / row.path), size, mode)
            rgb = np.stack([g] * 3, -1)
            cam, probs = class_activation_maps(model, x[None].to(device), int(row.label), "gradcam")
            campp, _ = class_activation_maps(model, x[None].to(device), int(row.label), "gradcam++")
            comparison[exp].append(overlay(rgb, campp[0]))
            for c, (img, title) in enumerate([(rgb, f"{classes[int(row.label)]} (p={probs[0, int(row.label)]:.2f})"),
                                              (overlay(rgb, cam[0]), "Grad-CAM"),
                                              (overlay(rgb, campp[0]), "Grad-CAM++")]):
                axes[r, c].imshow(img)
                axes[r, c].set_title(title, fontsize=8)
                axes[r, c].axis("off")
        fig.tight_layout()
        fig.savefig(out / f"gradcam_{exp}.png", dpi=300)
        plt.close(fig)

        # ---- pointing game on Dataset-1 stones
        hits, hits_pp, energy = [], [], []
        det_hits = []
        for img_path, lbl_path in det_items:
            gray = read_gray(img_path)
            h, w = gray.shape
            boxes = read_yolo_boxes(lbl_path, w, h)
            if len(boxes) == 0:
                continue
            sb = boxes * np.array([size / w, size / h, size / w, size / h], dtype=np.float32)
            g, x = prep(gray, size, mode)
            cam, _ = class_activation_maps(model, x[None].to(device), stone, "gradcam")
            campp, _ = class_activation_maps(model, x[None].to(device), stone, "gradcam++")
            hits.append(pointing_game_hit(cam[0], sb))
            hits_pp.append(pointing_game_hit(campp[0], sb))
            energy.append(energy_in_boxes(campp[0], sb))
            if has_detection(model):
                t = det_transforms(cfg, train=False)(image=gray, bboxes=boxes, labels=[0] * len(boxes))
                xd = to_tensor(build_input(t["image"], mode))
                db = np.array(t["bboxes"], dtype=np.float32).reshape(-1, 4)
                with torch.no_grad():
                    o = model(xd[None].to(device), "det")
                pb, ps = decode_centernet(o, cfg.model.det_stride, 1)[0]
                cx, cy = (pb[0, 0] + pb[0, 2]) / 2, (pb[0, 1] + pb[0, 3]) / 2
                det_hits.append(any(x1 - 8 <= cx <= x2 + 8 and y1 - 8 <= cy <= y2 + 8 for x1, y1, x2, y2 in db))
        scores[exp] = {"n_images": len(hits), "pointing_game_gradcam": float(np.mean(hits)),
                       "pointing_game_gradcam++": float(np.mean(hits_pp)),
                       "cam_energy_in_boxes": float(np.mean(energy))}
        if det_hits:
            scores[exp]["pointing_game_det_heatmap"] = float(np.mean(det_hits))
        print(exp, scores[exp])

    # ---- cross-model comparison figure
    if len(comparison) > 1:
        n = len(examples)
        fig, axes = plt.subplots(n, len(comparison), figsize=(2.4 * len(comparison), 2.4 * n))
        axes = np.atleast_2d(axes)
        if axes.shape[0] != n:
            axes = axes.T
        for c, (exp, ims) in enumerate(comparison.items()):
            for r, im in enumerate(ims):
                axes[r, c].imshow(im)
                axes[r, c].axis("off")
                if r == 0:
                    axes[r, c].set_title(exp, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "gradcam_comparison.png", dpi=300)
        plt.close(fig)

    save_json(scores, out / "pointing_game.json")
    pd.DataFrame(scores).T.to_csv(out / "pointing_game.csv")


if __name__ == "__main__":
    main()
