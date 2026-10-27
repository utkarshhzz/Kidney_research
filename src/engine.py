"""Training / evaluation engine."""
from __future__ import annotations

import copy
import itertools
import math
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .config import save_config
from .data.datasets import build_cls_loaders, build_det_loaders, class_weights
from .losses import CenterNetLoss, TaskWeighting, build_cls_loss
from .metrics import classification_metrics, detection_metrics
from .models import build_model, has_detection, param_groups
from .preprocessing import denormalize
from .utils import AverageMeter, get_logger, save_json
from . import viz


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
class ModelEMA:
    """Exponential moving average of weights (with warm-up of the decay)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / 2000))
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def _infinite(loader):
    """Re-iterate a DataLoader forever (fresh shuffling + augmentation every pass).
    NB: itertools.cycle would cache and replay the first pass's batches."""
    while True:
        yield from loader


def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    warm = cfg.train.warmup_epochs * steps_per_epoch
    total = cfg.train.epochs * steps_per_epoch
    floor = cfg.train.min_lr_ratio

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / max(warm, 1)
        if cfg.train.scheduler == "constant":
            return 1.0
        t = (step - warm) / max(total - warm, 1)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                          enabled=enabled and device.type == "cuda")


# ----------------------------------------------------------------------------
# inference
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict_cls(model, loader, device, amp: bool = True, tta_hflip: bool = False):
    model.eval()
    probs, labels, idxs = [], [], []
    for b in tqdm(loader, desc="predict", leave=False, ncols=80):
        x = b["image"].to(device, non_blocking=True)
        with autocast(device, amp):
            p = F.softmax(model(x, "cls")["logits"].float(), 1)
            if tta_hflip:
                p = (p + F.softmax(model(torch.flip(x, dims=[3]), "cls")["logits"].float(), 1)) / 2
        probs.append(p.cpu())
        labels.append(b["label"])
        idxs.append(b["index"])
    return torch.cat(probs).numpy(), torch.cat(labels).numpy(), torch.cat(idxs).numpy()


def decode_centernet(out: dict, stride: int, topk: int = 50):
    hm = torch.sigmoid(out["hm"].float())
    keep = (F.max_pool2d(hm, 3, 1, 1) == hm).float()
    hm = hm * keep
    b, _, h, w = hm.shape
    scores, inds = hm.view(b, -1).topk(min(topk, h * w))
    off = out["off"].float().view(b, 2, -1)
    wh = out["wh"].float().view(b, 2, -1)
    results = []
    for i in range(b):
        ind = inds[i]
        cx = (ind % w).float() + off[i, 0, ind]
        cy = (ind // w).float() + off[i, 1, ind]
        bw, bh = wh[i, 0, ind], wh[i, 1, ind]
        boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1) * stride
        results.append((boxes.cpu().numpy(), scores[i].cpu().numpy()))
    return results


@torch.no_grad()
def predict_det(model, loader, device, stride: int, topk: int, amp: bool = True, keep_images: int = 0):
    model.eval()
    preds, gts, images = [], [], []
    for b in tqdm(loader, desc="predict-det", leave=False, ncols=80):
        x = b["image"].to(device, non_blocking=True)
        with autocast(device, amp):
            out = model(x, "det")
        preds += decode_centernet(out, stride, topk)
        for j in range(len(x)):
            n = int(b["num_boxes"][j])
            gts.append(b["boxes"][j, :n].numpy())
            if len(images) < keep_images:
                images.append(denormalize(b["image"][j]))
    return preds, gts, images


# ----------------------------------------------------------------------------
# trainer
# ----------------------------------------------------------------------------
class Trainer:
    def __init__(self, cfg, run_dir: Path, resume: bool = False):
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "plots").mkdir(exist_ok=True)
        save_config(cfg, self.run_dir / "config.yaml")
        self.log = get_logger(f"train.{cfg.experiment.name}.{cfg.data.fold}", self.run_dir / "train.log")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = bool(cfg.train.amp) and self.device.type == "cuda"
        self.classes = list(cfg.data.classes)

        self.cls = build_cls_loaders(cfg)
        self.model = build_model(cfg).to(self.device)
        self.det_on = has_detection(self.model)
        self.det = build_det_loaders(cfg) if self.det_on else None

        tr = self.cls["frame"]
        w = class_weights(tr.loc[tr.split == "train", "label"].values, len(self.classes), cfg.loss.class_weighting)
        self.cls_loss = build_cls_loss(cfg, w).to(self.device)
        self.det_loss = CenterNetLoss(cfg) if self.det_on else None
        self.task_w = TaskWeighting(cfg.loss.mtl_weighting, cfg.loss.mtl_fixed_det_weight).to(self.device)

        groups = param_groups(self.model, cfg.train.backbone_lr, cfg.train.lr, cfg.train.weight_decay)
        if self.det_on and cfg.loss.mtl_weighting == "uncertainty":
            groups.append({"params": list(self.task_w.parameters()), "lr": cfg.train.lr, "weight_decay": 0.0})
        if cfg.train.optimizer == "adamw":
            self.opt = torch.optim.AdamW(groups)
        elif cfg.train.optimizer == "sgd":
            self.opt = torch.optim.SGD(groups, momentum=0.9, nesterov=True)
        else:
            raise ValueError(cfg.train.optimizer)
        self.det_only = bool(cfg.train.det_only)
        if self.det_only and not self.det_on:
            raise ValueError("train.det_only=true requires model.det_enabled=true")
        self.steps_per_epoch = len(self.det["train"] if self.det_only else self.cls["train"])
        if cfg.train.max_steps_per_epoch:
            self.steps_per_epoch = min(self.steps_per_epoch, int(cfg.train.max_steps_per_epoch))
        self.sched = build_scheduler(self.opt, cfg, self.steps_per_epoch)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.ema = ModelEMA(self.model, cfg.train.ema_decay) if cfg.train.ema else None

        if self.det_only and cfg.train.monitor != "val_det_map50":
            raise ValueError("det_only runs must use train.monitor=val_det_map50")
        self.start_epoch, self.best, self.bad_epochs, self.history = 0, -float("inf"), 0, []
        if resume and (self.run_dir / "last.pt").exists():
            self._load_last()

        n_params = sum(p.numel() for p in self.model.parameters())
        self.log.info(f"device={self.device} amp={self.amp} params={n_params / 1e6:.2f}M det={self.det_on} "
                      f"train={len(self.cls['train'].dataset)} val={len(self.cls['val'].dataset)} "
                      f"test={len(self.cls['test'].dataset)}")

    # ------------------------------------------------------------------ io
    def _eval_model(self):
        return self.ema.module if self.ema is not None else self.model

    def _save_last(self, epoch):
        torch.save({
            "epoch": epoch, "model": self.model.state_dict(),
            "ema": self.ema.module.state_dict() if self.ema else None,
            "ema_updates": self.ema.updates if self.ema else 0,
            "opt": self.opt.state_dict(), "sched": self.sched.state_dict(), "scaler": self.scaler.state_dict(),
            "task_w": self.task_w.state_dict(), "best": self.best, "bad_epochs": self.bad_epochs,
            "history": self.history,
        }, self.run_dir / "last.pt")

    def _load_last(self):
        ck = torch.load(self.run_dir / "last.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        if self.ema and ck["ema"] is not None:
            self.ema.module.load_state_dict(ck["ema"])
            self.ema.updates = ck["ema_updates"]
        self.opt.load_state_dict(ck["opt"])
        self.sched.load_state_dict(ck["sched"])
        self.scaler.load_state_dict(ck["scaler"])
        self.task_w.load_state_dict(ck["task_w"])
        self.start_epoch, self.best = ck["epoch"] + 1, ck["best"]
        self.bad_epochs, self.history = ck["bad_epochs"], ck["history"]
        self.log.info(f"resumed from epoch {ck['epoch']}")

    # --------------------------------------------------------------- train
    def _train_epoch(self, epoch: int) -> dict:
        cfg = self.cfg
        self.model.train()
        meters = {k: AverageMeter() for k in ("loss", "cls_loss", "det_loss")}
        det_iter = _infinite(self.det["train"]) if self.det_on else None
        main_loader = self.det["train"] if self.det_only else self.cls["train"]
        pbar = tqdm(itertools.islice(main_loader, self.steps_per_epoch), total=self.steps_per_epoch,
                    desc=f"epoch {epoch}", ncols=100, leave=False)
        for step, b in enumerate(pbar):
            x = b["image"].to(self.device, non_blocking=True)
            with autocast(self.device, self.amp):
                if self.det_only:
                    db = {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
                          for k, v in b.items()}
                    l_det, _ = self.det_loss(self.model(x, "det"), db)
                    l_cls = torch.zeros((), device=self.device)
                    loss = l_det
                elif self.det_on:
                    y = b["label"].to(self.device, non_blocking=True)
                    l_cls = self.cls_loss(self.model(x, "cls")["logits"], y)
                    db = {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
                          for k, v in next(det_iter).items()}
                    l_det, _ = self.det_loss(self.model(db["image"], "det"), db)
                    loss = self.task_w(l_cls, l_det)
                else:
                    y = b["label"].to(self.device, non_blocking=True)
                    l_cls = self.cls_loss(self.model(x, "cls")["logits"], y)
                    l_det = torch.zeros((), device=self.device)
                    loss = l_cls
            if not torch.isfinite(loss):
                self.log.info(f"non-finite loss at step {step}, skipping batch")
                self.opt.zero_grad(set_to_none=True)
                continue
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if cfg.train.grad_clip:
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.train.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.sched.step()
            if self.ema:
                self.ema.update(self.model)
            n = len(x)
            meters["loss"].update(loss.item(), n)
            meters["cls_loss"].update(l_cls.item(), n)
            meters["det_loss"].update(l_det.item(), n)
            if step % cfg.train.log_every == 0:
                pbar.set_postfix(loss=f"{meters['loss'].avg:.4f}")
        out = {f"train_{k}": m.avg for k, m in meters.items()}
        if not self.det_on:
            out.pop("train_det_loss")
        if self.det_only:
            out.pop("train_cls_loss")
        out.update(self.task_w.weights() if self.det_on and not self.det_only else {})
        out["lr"] = self.opt.param_groups[0]["lr"]
        return out

    def _validate(self) -> dict:
        m = self._eval_model()
        out = {}
        if not self.det_only:
            probs, y, _ = predict_cls(m, self.cls["val"], self.device, self.amp)
            cm = classification_metrics(y, probs, self.classes, self.cfg.eval.ece_bins)
            out = {"val_accuracy": cm["accuracy"], "val_macro_f1": cm["macro_f1"],
                   "val_balanced_accuracy": cm["balanced_accuracy"], "val_nll": cm["nll"]}
        if self.det_on:
            preds, gts, _ = predict_det(m, self.det["val"], self.device, self.cfg.model.det_stride,
                                        self.cfg.eval.det_topk, self.amp)
            out["val_det_map50"] = detection_metrics(preds, gts, [0.5])["mAP@0.5"]
            if not self.det_only:
                out["val_combined"] = 0.5 * (out["val_macro_f1"] + out["val_det_map50"])
        return out

    def fit(self) -> dict:
        cfg = self.cfg
        t0 = time.time()
        for epoch in range(self.start_epoch, cfg.train.epochs):
            te = time.time()
            row = {"epoch": epoch, **self._train_epoch(epoch), **self._validate()}
            row["epoch_time_s"] = time.time() - te
            self.history.append(row)
            score = row[cfg.train.monitor]
            improved = score > self.best
            if improved:
                self.best, self.bad_epochs = score, 0
                torch.save({"model": self._eval_model().state_dict(), "epoch": epoch, "score": score,
                            "config": cfg.to_dict()}, self.run_dir / "best.pt")
            else:
                self.bad_epochs += 1
            msg = " ".join(f"{k}={v:.4f}" for k, v in row.items() if isinstance(v, float) and k != "lr")
            self.log.info(f"ep {epoch:03d} {msg} {'*' if improved else ''}")
            pd.DataFrame(self.history).to_csv(self.run_dir / "history.csv", index=False)
            self._save_last(epoch)
            if self.bad_epochs >= cfg.train.early_stopping_patience:
                self.log.info(f"early stopping at epoch {epoch}")
                break
        train_time = time.time() - t0
        hist = pd.DataFrame(self.history)
        if len(hist):
            viz.plot_history(hist, self.run_dir / "plots" / "curves.png")
        results = self.test()
        results["train_time_s"] = train_time
        results["epochs_run"] = len(self.history)
        save_json(results["summary"], self.run_dir / "summary.json")
        if (self.run_dir / "last.pt").exists():
            (self.run_dir / "last.pt").unlink()
        if not cfg.experiment.keep_checkpoint and (self.run_dir / "best.pt").exists():
            (self.run_dir / "best.pt").unlink()
        (self.run_dir / "DONE").write_text("ok")
        return results

    # ---------------------------------------------------------------- test
    def test(self) -> dict:
        model, epoch = load_run_model(self.run_dir, self.device)
        return evaluate_run(self.cfg, model, self.cls, self.det, self.run_dir, self.device, self.amp,
                            best_epoch=epoch, log=self.log)


def load_run_model(run_dir: Path, device):
    """Rebuild a model from <run_dir>/best.pt (no ImageNet download needed)."""
    from .config import Config

    ck = torch.load(Path(run_dir) / "best.pt", map_location=device, weights_only=False)
    cfg = Config.wrap(copy.deepcopy(ck["config"]))
    cfg.model.pretrained = False
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), ck["epoch"]


def evaluate_run(cfg, model, cls_loaders, det_loaders, run_dir: Path, device, amp: bool,
                 best_epoch: int | None = None, log=None) -> dict:
    """Full test-time evaluation; writes metrics JSON, predictions CSV and plots."""
    run_dir = Path(run_dir)
    classes = list(cfg.data.classes)
    plots = run_dir / "plots"
    plots.mkdir(exist_ok=True)
    summary = {"experiment": cfg.experiment.name, "fold": cfg.data.fold, "protocol": cfg.data.split_protocol,
               "best_epoch": best_epoch}
    for split in (() if cfg.train.det_only else ("val", "test")):
        probs, y, idx = predict_cls(model, cls_loaders[split], device, amp, tta_hflip=cfg.eval.tta_hflip)
        m = classification_metrics(y, probs, classes, cfg.eval.ece_bins)
        save_json(m, run_dir / f"{split}_metrics.json")
        df = cls_loaders[split].dataset.df.iloc[idx].reset_index(drop=True)
        pred_df = pd.DataFrame({"path": df["path"], "label": y, "pred": probs.argmax(1)})
        for i, c in enumerate(classes):
            pred_df[f"prob_{c}"] = probs[:, i]
        for col in ("group", "aug_id"):
            if col in df:
                pred_df[col] = df[col].values
        pred_df.to_csv(run_dir / f"{split}_predictions.csv", index=False)
        if split == "test":
            viz.plot_confusion(m["confusion_matrix"], classes, plots / "confusion_matrix.png")
            viz.plot_roc(y, probs, classes, plots / "roc.png")
            viz.plot_reliability(y, probs, plots / "reliability.png", cfg.eval.ece_bins)
            summary.update({f"test_{k}": v for k, v in m.items() if isinstance(v, float)})
            if "stone_vs_rest" in m:
                summary.update({f"test_stone_{k}": v for k, v in m["stone_vs_rest"].items()})
            if log:
                log.info(f"TEST acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
                         f"AUC={m.get('macro_auc', float('nan')):.4f} ECE={m['ece']:.4f}")
    if det_loaders is not None and has_detection(model):
        for split in ("val", "test"):
            preds, gts, imgs = predict_det(model, det_loaders[split], device, cfg.model.det_stride,
                                           cfg.eval.det_topk, amp, keep_images=12 if split == "test" else 0)
            dm = detection_metrics(preds, gts, cfg.eval.det_iou_thresholds, cfg.eval.det_score_threshold)
            save_json(dm, run_dir / f"det_{split}_metrics.json")
            if split == "test":
                viz.draw_detections(imgs, preds[:len(imgs)], gts[:len(imgs)], plots / "detections.png",
                                    cfg.eval.det_score_threshold)
                summary.update({f"det_{k}": v for k, v in dm.items() if isinstance(v, float)})
                if log:
                    log.info(f"TEST det mAP50={dm['mAP@0.5']:.4f} mAP50-95={dm['mAP@0.5:0.95']:.4f}")
    return {"summary": summary}
