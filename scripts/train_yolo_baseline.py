"""Optional detection baseline: Ultralytics YOLOv8 on Dataset 1.

    pip install ultralytics
    python scripts/train_yolo_baseline.py --model yolov8n.pt
    python scripts/train_yolo_baseline.py --model yolov8s.pt

Results land in results/runs/yolo_<model>/metrics.json with the same keys the
aggregator uses, so YOLO appears in the detection table automatically.
Note: YOLO uses its own (heavier) augmentation/recipe; we report it as an
external reference point, not a controlled comparison.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from src.config import load_config, resolve_path  # noqa: E402
from src.utils import save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--output-root", default=None, help="default: experiment.output_root (results/runs)")
    args = ap.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("pip install ultralytics  (optional dependency)")

    cfg = load_config()
    det_root = resolve_path(cfg.data.root) / cfg.data.det_dir
    name = f"yolo_{Path(args.model).stem}"
    run_dir = resolve_path(args.output_root or cfg.experiment.output_root) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = run_dir / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(det_root), "train": "train/images", "val": "valid/images", "test": "test/images",
        "nc": 1, "names": ["stone"]}))

    model = YOLO(args.model)
    model.train(data=str(data_yaml), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
                project=str(run_dir), name="train", seed=cfg.experiment.seed, exist_ok=True,
                single_cls=True, patience=20)
    best = YOLO(str(run_dir / "train" / "weights" / "best.pt"))
    m = best.val(data=str(data_yaml), split="test", imgsz=args.imgsz, project=str(run_dir), name="test",
                 exist_ok=True)
    p, r = float(m.box.mp), float(m.box.mr)
    save_json({"label": f"YOLOv8-{Path(args.model).stem[-1].upper()} (Ultralytics)",
               "det_mAP@0.5": float(m.box.map50), "det_mAP@0.5:0.95": float(m.box.map),
               "det_precision@0.5": p, "det_recall@0.5": r,
               "det_f1@0.5": 2 * p * r / max(p + r, 1e-9)}, run_dir / "metrics.json")
    print(open(run_dir / "metrics.json").read())


if __name__ == "__main__":
    main()
