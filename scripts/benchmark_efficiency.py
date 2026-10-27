"""Parameters, FLOPs and latency for every distinct architecture.

    python scripts/benchmark_efficiency.py            # GPU if available
    python scripts/benchmark_efficiency.py --device cpu

FLOPs are counted with torch.utils.flop_counter (1 multiply-add = 2 FLOPs),
latency is batch-1 median over --iters runs after warm-up. Run on the same
machine for all models and report the hardware in the paper.
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from src.config import load_config, resolve_path  # noqa: E402
from src.experiments import EXPERIMENTS, experiment_opts  # noqa: E402
from src.models import build_model, has_detection  # noqa: E402
from src.reporting import to_latex  # noqa: E402

SELECTED = ["kidneymtl_main", "a_no_det", "b_resnet50", "b_densenet121", "b_efficientnet_b3",
            "b_convnext_tiny", "b_vit_small", "b_vgg19", "b_cdr_vgg19"]


def flops(model, x, task):
    with FlopCounterMode(display=False) as fc:
        model(x, task)
    return fc.get_total_flops()


@torch.no_grad()
def latency(model, x, task, warmup, iters, device):
    for _ in range(warmup):
        model(x, task)
    times = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        model(x, task)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--experiments", nargs="*", default=SELECTED)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", default="results/tables")
    args = ap.parse_args()

    device = torch.device(args.device)
    rows = []
    for name in args.experiments:
        cfg = load_config(None, experiment_opts(name) + ["model.pretrained=false"])
        model = build_model(cfg).to(device).eval()
        s = cfg.data.cls_img_size
        x = torch.randn(1, 3, s, s, device=device)
        total = sum(p.numel() for p in model.parameters())
        det_params = 0
        if has_detection(model):
            det_params = sum(p.numel() for m in (model.neck, model.det_head) for p in m.parameters())
        row = {"experiment": name, "label": EXPERIMENTS[name]["label"], "input": s,
               "params_M": total / 1e6, "cls_path_params_M": (total - det_params) / 1e6,
               "cls_GFLOPs": flops(model, x, "cls") / 1e9,
               "cls_latency_ms": latency(model, x, "cls", args.warmup, args.iters, device)}
        if has_detection(model):
            d = cfg.data.det_img_size
            xd = torch.randn(1, 3, d, d, device=device)
            row["det_GFLOPs"] = flops(model, xd, "det") / 1e9
            row["det_latency_ms"] = latency(model, xd, "det", args.warmup, args.iters, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                model(torch.randn(32, 3, s, s, device=device), "cls")
            row["infer_mem_bs32_GB"] = torch.cuda.max_memory_allocated() / 1e9
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    hw = torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor()
    df["hardware"] = hw
    df.to_csv(out / "efficiency.csv", index=False)
    cols = ["label", "params_M", "cls_GFLOPs", "cls_latency_ms"]
    tex = df[cols].rename(columns={"label": "Model", "params_M": "Params (M)", "cls_GFLOPs": "GFLOPs",
                                   "cls_latency_ms": "Latency (ms)"})
    for c in ("Params (M)", "GFLOPs", "Latency (ms)"):
        tex[c] = tex[c].map(lambda v: f"{v:.2f}")
    to_latex(tex, out / "efficiency.tex", f"Model complexity (batch 1, {hw}).", "tab:efficiency")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
