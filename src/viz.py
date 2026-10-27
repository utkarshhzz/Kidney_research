"""Publication figures (matplotlib, 300 dpi)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from .metrics import reliability_curve  # noqa: E402

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
PALETTE = ["#2a6fdb", "#e0533d", "#27a35a", "#9b51e0", "#f2a516", "#555555"]


def plot_history(history: pd.DataFrame, path: Path) -> None:
    cols = [c for c in ("train_loss", "train_cls_loss", "train_det_loss") if c in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for i, c in enumerate(cols):
        axes[0].plot(history["epoch"], history[c], label=c.replace("train_", ""), color=PALETTE[i])
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("training loss")
    axes[0].legend(frameon=False)
    mcols = [c for c in ("val_accuracy", "val_macro_f1", "val_det_map50") if c in history]
    for i, c in enumerate(mcols):
        axes[1].plot(history["epoch"], history[c], label=c.replace("val_", ""), color=PALETTE[i])
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation metric")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion(cm, class_names, path: Path, normalize: bool = True) -> None:
    cm = np.asarray(cm, dtype=np.float64)
    show = cm / cm.sum(1, keepdims=True).clip(min=1) if normalize else cm
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(show, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)), class_names, rotation=30)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            txt = f"{show[i, j]:.2f}\n({int(cm[i, j])})" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if show[i, j] > 0.6 * show.max() else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_roc(y, probs, class_names, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    for i, c in enumerate(class_names):
        yb = (np.asarray(y) == i).astype(int)
        if 0 < yb.sum() < len(yb):
            fpr, tpr, _ = roc_curve(yb, probs[:, i])
            ax.plot(fpr, tpr, label=c, color=PALETTE[i % len(PALETTE)])
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=0.8)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_reliability(y, probs, path: Path, n_bins: int = 15) -> None:
    centers, accs, counts = reliability_curve(np.asarray(probs), np.asarray(y), n_bins)
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=0.8, label="perfect calibration")
    ax.plot(centers, accs, "o-", color=PALETTE[0], label="model")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def draw_detections(images, preds, gts, path: Path, score_thr: float = 0.3, ncols: int = 4) -> None:
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    for ax in np.atleast_1d(axes).flat:
        ax.axis("off")
    for ax, img, (boxes, scores), gt in zip(np.atleast_1d(axes).flat, images, preds, gts):
        ax.imshow(img[..., 0], cmap="gray")
        for x1, y1, x2, y2 in gt:
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="#27a35a", lw=1.2))
        for (x1, y1, x2, y2), s in zip(boxes, scores):
            if s >= score_thr:
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="#e0533d", lw=1.0, ls="--"))
                ax.text(x1, y1 - 2, f"{s:.2f}", color="#e0533d", fontsize=6)
    fig.suptitle("green = ground truth, red dashed = prediction", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
