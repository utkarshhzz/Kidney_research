"""Evaluation metrics and statistical tests."""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, log_loss, matthews_corrcoef, precision_recall_fscore_support,
                             roc_auc_score)


# ----------------------------------------------------------------------------
# classification
# ----------------------------------------------------------------------------
def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_curve(probs: np.ndarray, y: np.ndarray, n_bins: int = 15):
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    centers, accs, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            centers.append(conf[m].mean())
            accs.append(correct[m].mean())
            counts.append(int(m.sum()))
    return np.array(centers), np.array(accs), np.array(counts)


def classification_metrics(y: np.ndarray, probs: np.ndarray, class_names: list[str],
                           ece_bins: int = 15, positive: str = "Stone") -> dict:
    y = np.asarray(y)
    probs = np.asarray(probs, dtype=np.float64)
    pred = probs.argmax(1)
    k = len(class_names)
    labels = list(range(k))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y, pred, labels=labels)
    spec = []
    for c in labels:
        tn = cm.sum() - cm[c, :].sum() - cm[:, c].sum() + cm[c, c]
        fp = cm[:, c].sum() - cm[c, c]
        spec.append(tn / max(tn + fp, 1))
    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_precision": float(np.mean(p)),
        "macro_recall": float(np.mean(r)),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
        "macro_specificity": float(np.mean(spec)),
        "cohen_kappa": cohen_kappa_score(y, pred),
        "mcc": matthews_corrcoef(y, pred),
        "ece": expected_calibration_error(probs, y, ece_bins),
        "brier": float(np.mean(np.sum((probs - np.eye(k)[y]) ** 2, axis=1))),
        "nll": log_loss(y, np.clip(probs, 1e-7, 1), labels=labels),
        "n": int(len(y)),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                "specificity": float(spec[i]), "support": int(s[i])}
            for i, c in enumerate(class_names)
        },
    }
    if len(np.unique(y)) == k:
        out["macro_auc"] = roc_auc_score(y, probs, multi_class="ovr", average="macro", labels=labels)
        for i, c in enumerate(class_names):
            out["per_class"][c]["auc"] = roc_auc_score((y == i).astype(int), probs[:, i])
    if positive in class_names:  # stone-vs-rest view, comparable to binary stone papers
        pi = class_names.index(positive)
        yb, pb = (y == pi).astype(int), (pred == pi).astype(int)
        tp, fn = int(((pb == 1) & (yb == 1)).sum()), int(((pb == 0) & (yb == 1)).sum())
        tn, fp = int(((pb == 0) & (yb == 0)).sum()), int(((pb == 1) & (yb == 0)).sum())
        out["stone_vs_rest"] = {
            "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "ppv": tp / max(tp + fp, 1), "npv": tn / max(tn + fn, 1),
            "auc": roc_auc_score(yb, probs[:, pi]) if 0 < yb.sum() < len(yb) else float("nan"),
        }
    return {k_: (float(v) if isinstance(v, (np.floating, float)) else v) for k_, v in out.items()}


# ----------------------------------------------------------------------------
# detection (COCO-style AP, single class)
# ----------------------------------------------------------------------------
def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _match(preds, gts, iou_t):
    """Greedy score-ordered matching. Returns (scores, is_tp) over all predictions, n_gt."""
    scores, tps, n_gt = [], [], 0
    for (pb, ps), gb in zip(preds, gts):
        n_gt += len(gb)
        order = np.argsort(-ps)
        pb, ps = pb[order], ps[order]
        ious = box_iou_np(pb, gb)
        used = np.zeros(len(gb), bool)
        for i in range(len(pb)):
            tp = False
            if len(gb):
                cand = np.where(~used, ious[i], -1)
                j = int(cand.argmax())
                if cand[j] >= iou_t:
                    used[j] = True
                    tp = True
            scores.append(ps[i])
            tps.append(tp)
    return np.array(scores), np.array(tps, bool), n_gt


def average_precision(preds, gts, iou_t: float) -> float:
    scores, tps, n_gt = _match(preds, gts, iou_t)
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    tps = tps[order]
    tp_c = np.cumsum(tps)
    fp_c = np.cumsum(~tps)
    recall = tp_c / n_gt
    precision = tp_c / np.maximum(tp_c + fp_c, 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # monotone envelope
    rs = np.linspace(0, 1, 101)
    ap = [precision[np.searchsorted(recall, r, side="left")] if r <= recall[-1] else 0.0 for r in rs]
    return float(np.mean(ap))


def detection_metrics(preds, gts, iou_thresholds, score_threshold: float = 0.3) -> dict:
    """preds: list of (boxes Nx4, scores N); gts: list of Mx4 arrays (same image order)."""
    aps = {f"AP@{t:.2f}": average_precision(preds, gts, t) for t in iou_thresholds}
    out = {"mAP@0.5": aps.get("AP@0.50", average_precision(preds, gts, 0.5)),
           "mAP@0.5:0.95": float(np.nanmean(list(aps.values()))),
           "AP@0.75": aps.get("AP@0.75", float("nan"))}
    filt = [(b[s >= score_threshold], s[s >= score_threshold]) for b, s in preds]
    _, tps, n_gt = _match(filt, gts, 0.5)
    tp = int(tps.sum())
    n_pred = len(tps)
    prec, rec = tp / max(n_pred, 1), tp / max(n_gt, 1)
    out.update({"precision@0.5": prec, "recall@0.5": rec,
                "f1@0.5": 2 * prec * rec / max(prec + rec, 1e-9),
                "score_threshold": score_threshold, "n_images": len(gts), "n_gt_boxes": n_gt})
    img_hit = [len(g) > 0 and len(b) > 0 for (b, _), g in zip(filt, gts)]
    out["image_level_stone_detection_rate"] = float(np.mean(img_hit)) if img_hit else float("nan")
    return out


# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------
def bootstrap_ci(y, probs, metric_fn, iters: int = 2000, alpha: float = 0.05, seed: int = 0,
                 groups: np.ndarray | None = None):
    """Percentile bootstrap CI. If ``groups`` is given, resample whole groups
    (cluster bootstrap) so correlated near-duplicate slices are not treated as
    independent evidence."""
    rng = np.random.default_rng(seed)
    y, probs = np.asarray(y), np.asarray(probs)
    vals = []
    if groups is None:
        n = len(y)
        for _ in range(iters):
            idx = rng.integers(0, n, n)
            try:
                vals.append(metric_fn(y[idx], probs[idx]))
            except ValueError:
                continue
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        members = {g: np.where(groups == g)[0] for g in uniq}
        for _ in range(iters):
            gs = rng.choice(uniq, len(uniq), replace=True)
            idx = np.concatenate([members[g] for g in gs])
            try:
                vals.append(metric_fn(y[idx], probs[idx]))
            except ValueError:
                continue
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact McNemar test on paired per-sample correctness."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01 = int((a & ~b).sum())  # A right, B wrong
    n10 = int((~a & b).sum())
    n = n01 + n10
    p = 1.0 if n == 0 else stats.binomtest(min(n01, n10), n, 0.5).pvalue
    return {"a_only_correct": n01, "b_only_correct": n10, "p_value": float(p)}


def holm_correction(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def mean_std(values) -> tuple[float, float]:
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], dtype=np.float64)
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
