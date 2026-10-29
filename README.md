# KidneyMTL — Leakage-Aware Multi-Task Learning for Kidney Disease Classification and Stone Localisation in CT

Code for the paper (4-page IEEE version for Overleaf in [`paper/overleaf/main.tex`](paper/overleaf/main.tex),
long draft in [`paper/main.tex`](paper/main.tex)). One shared backbone learns
**4-class kidney CT classification** (Normal / Cyst / Tumor / Stone, CT-KIDNEY dataset) and
**kidney-stone localisation** (bounding boxes, Roboflow kidney-stone dataset) simultaneously, and is evaluated
with a **leakage-free, near-duplicate-aware cross-validation protocol**.

## Why this paper is different (read this first)

The reference paper (CDR-VGG19, Devi et al., *Appl. Comput. Intell. Soft Comput.* 2025) reports 99.61 % accuracy,
but it **augments every image 21× and then splits 80:10:10**. Augmented copies of the same image then sit in both
train and test. Our split audit ([`data/splits/leakage_report.json`](data/splits/leakage_report.json),
produced by `scripts/make_splits.py`) shows how big this problem is on CT-KIDNEY:

| Protocol | Test images with a near-duplicate (≤ 4/64 pHash bits) in train | Median distance to nearest training image |
|---|---|---|
| Random image-level split (common practice) | **99.3 %** | **0 bits** (identical hash) |
| Augment-then-split (reference paper) | 98.6 % of test *source* images are also in train | — |
| **Grouped split (ours)** | **0 %** | 10.4 bits |

The dataset also contains **517 sets of byte-identical duplicate files (1,034 images)**. Accuracy above 99 % on
this dataset therefore mostly measures memorisation. The paper's contributions are:

1. **A leakage audit and a grouped evaluation protocol** (perceptual-hash near-duplicate groups → StratifiedGroupKFold),
   and a controlled experiment that re-trains the same models under all three protocols to *quantify* the inflation.
2. **KidneyMTL**: an EfficientNet-B3 backbone with a CBAM classification head and a CenterNet-style stone detector on
   an FPN neck, trained on two partially-labelled datasets with homoscedastic-uncertainty task weighting.
3. **A tri-channel input** `[CT, CLAHE(CT), multi-level contour map]` that keeps the reference paper's contour idea
   without throwing away the image.
4. **A complete evaluation**: 5-fold CV (mean ± std), pooled out-of-fold predictions with cluster-bootstrap 95 % CIs,
   McNemar tests with Holm correction, calibration (ECE, reliability diagrams), quantitative explainability
   (Grad-CAM pointing game against stone boxes), cross-dataset/cross-view external validation, and an efficiency table.
   The reference model is re-implemented and re-evaluated under the same protocol.

> Expect honest numbers lower than 99.6 % under the grouped protocol. That is the point. Reviewers increasingly
> reject near-perfect results on this dataset, and the leakage experiment shows *why* the earlier numbers are inflated.

## Repository layout

```
configs/default.yaml        every hyperparameter (one file; experiments = overrides)
src/
  config.py                 YAML config + CLI overrides
  preprocessing.py          input representations (gray, CLAHE, contour, Sobel, Canny, Otsu, tri)
  augmentation.py           online train-only augmentation; offline variants for the leakage replication
  data/splits.py            pHash grouping, fold construction, leakage audit
  data/datasets.py          classification + CenterNet detection datasets and loaders
  models/kidney_mtl.py      KidneyMTL (backbone + CBAM cls head + FPN + CenterNet head)
  models/base_models.py     timm baselines (ResNet, DenseNet, EfficientNet, ConvNeXt, ViT, VGG)
  models/cdr_vgg19.py       re-implementation of the reference CDR-VGG19
  losses.py                 focal loss, CenterNet focal + CIoU/GIoU, uncertainty task weighting
  engine.py                 training loop (AMP, EMA, warmup+cosine, early stopping, resume), evaluation
  metrics.py                classification/detection metrics, ECE, bootstrap CIs, McNemar, Holm
  explain.py                Grad-CAM / Grad-CAM++, pointing game
  experiments.py            registry of every experiment in the paper
scripts/
  make_splits.py            build folds + leakage report (already run; splits are committed)
  smoke_test.py             ~5-minute end-to-end check on a new machine
  train.py                  train + test one experiment on one fold
  run_experiments.py        run the whole experiment matrix (resumable)
  aggregate_results.py      tables (CSV + LaTeX), statistics, figures, results/RESULTS.md
  explain.py                Grad-CAM figures + pointing-game scores
  external_validation.py    classifiers on the unseen Dataset-1 test split
  benchmark_efficiency.py   params / FLOPs / latency
  evaluate.py               re-evaluate a finished run (e.g. without TTA)
  train_yolo_baseline.py    optional YOLOv8 detection reference
  download_data.py          Kaggle download + extraction + layout check (Drive zip cache on Colab)
  package_results.py        zip all results (no weights) for download
  import_results.py         merge downloaded result zips into results/ on the local machine
paper/                      overleaf/main.tex (4-page IEEE paper) + long draft + bibliography
data/splits/                exact folds used in the paper (committed for reproducibility)
notebooks/KidneyMTL_Colab.ipynb   <- the notebook to train everything on Google Colab
notebooks/ (others)         earlier notebooks (03_data_augmentation is DEPRECATED, see its first cell)
```

## Training on Google Colab (recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/utkarshhzz/Kidney_research/blob/main/notebooks/KidneyMTL_Colab.ipynb)

1. Open [`notebooks/KidneyMTL_Colab.ipynb`](notebooks/KidneyMTL_Colab.ipynb) with the badge above, then choose
   `Runtime → Change runtime type → GPU`.
2. Add a **new** Kaggle token as Colab Secrets `KAGGLE_USERNAME` / `KAGGLE_KEY` (key icon in the sidebar), or put
   `kaggle.json` in `MyDrive/Kidney_research/`.
3. Run sections 1–3 (setup, data, checks), then section 4 with plan `paper_3folds` (the 24 runs behind the
   4-page paper, ~16 T4-hours; run 4.2 again in each new session). Larger plans: `tier1_3folds` → `tier1` → `all`.
4. Everything is saved to **Google Drive** (`MyDrive/Kidney_research/results`). After a disconnect, run sections 1, 2
   and 4.2 again: finished runs are skipped. Checkpoints live on Colab's local disk, so Drive only receives results
   plus the final `best.pt` of the main model (all folds) and of the baselines (fold 0).
5. Section 5 builds all tables, figures, Grad-CAM and the efficiency table. Section 6 zips everything (without
   model weights) and **downloads it automatically** (a copy also goes to `MyDrive/Kidney_research/downloads/`).
6. On your own computer, inside this project folder (no GPU or PyTorch needed):
   ```bash
   python scripts/import_results.py results_YYYYMMDD_HHMM.zip   # merges into results/
   python scripts/aggregate_results.py                          # rebuilds results/RESULTS.md + tables
   ```

Colab-specific engineering in the code: the classification and detection losses are back-propagated separately
(identical gradients, about half the peak GPU memory, so the multi-task model fits a 16 GB T4). Runs that crash with
CUDA out-of-memory are retried automatically with half the batch size. `--hours` stops starting new runs before the
session limit. Every run also writes its external-validation metrics, so no table needs checkpoints.

## 1. Setup (alternative: your own GPU machine)

```bash
git clone https://github.com/utkarshhzz/Kidney_research.git
cd Kidney_research
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt
```

## 2. Data

Download both Kaggle datasets into `data/`, giving exactly this layout (the same as the original project):

```
data/dataset1/{train,valid,test}/{images,labels}        # kaggle: safurahajiheidari/kidney-stone-images (YOLO format)
data/dataset2/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/{Cyst,Normal,Stone,Tumor}/
                                                        # kaggle: nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone
```

```bash
kaggle datasets download -d safurahajiheidari/kidney-stone-images -p data && unzip -q data/kidney-stone-images.zip -d data/dataset1
kaggle datasets download -d nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone -p data && unzip -q data/ct-kidney-dataset-normal-cyst-tumor-and-stone.zip -d data/dataset2
```

The folds in `data/splits/` are already built. Only run `python scripts/make_splits.py` if you change the data or
the grouping. It reuses cached hashes, and `--method hybrid` adds CNN-embedding links.

## 3. Run

```bash
python scripts/smoke_test.py                 # ALWAYS first: ~5 min on CPU, catches environment problems
python scripts/run_experiments.py --dry-run  # shows the plan
python scripts/run_experiments.py --tier1    # minimum publishable set (10 experiments)
python scripts/run_experiments.py            # everything else (finished runs are skipped)
```

* Runs are **resumable**. If the machine restarts, run the same command again; finished runs are skipped and
  interrupted runs continue from `last.pt`. On Colab, an interrupted run restarts because its `last.pt` was on local disk.
* A crashed run writes its traceback to `results/runs/<experiment>/fold<k>/error.log`; the queue continues.
* Run single things with `python scripts/train.py --name kidneymtl_main --fold 0`, or with
  `scripts/run_experiments.py --only b_resnet50 --folds 0 1`.
* Out of GPU memory → append `--opts train.batch_size=16 train.det_batch_size=8`. Slow data loading → `data.num_workers=8`.
* **Time budget**: check `epoch_time_s` in `results/runs/kidneymtl_main/fold0/history.csv` after the first run.
  A full run is ≤ 40 epochs (early stopping usually ends it sooner). The `aug_then_split` runs see 5× more images per
  epoch, so they get 3 folds only.

### Experiment matrix (`src/experiments.py`)

| Group | Experiments | Folds | Purpose |
|---|---|---|---|
| main | `kidneymtl_main` | 5 | proposed model |
| baselines | ResNet-50, DenseNet-121, EfficientNet-B3, ConvNeXt-T, ViT-S/16, VGG-19, **CDR-VGG19 (reference, re-implemented)** | 5 | standard fine-tuning under the same protocol |
| ablations | w/o detection task, w/o CBAM, SE instead of CBAM, input gray / CLAHE / contour-only, CE instead of focal, fixed task weights, GIoU instead of CIoU, ReLU head, ResNet-50 / ConvNeXt-T backbones | 3 | one change each |
| detection | detection-only model | 1 | does multi-task learning help localisation? |
| leakage | KidneyMTL, ResNet-50 and CDR-VGG19 under the *random* and *augment-then-split* protocols | 5 / 3 | quantifies the inflation |

`--tier1` = main, CDR-VGG19, ResNet-50, EfficientNet-B3, ConvNeXt-T, w/o detection, gray input, and the three
key leakage runs. Run the rest when time allows.

## 4. Analyse (after training)

```bash
python scripts/aggregate_results.py                       # all tables + stats + figures -> results/
python scripts/explain.py --experiments kidneymtl_main b_resnet50 b_cdr_vgg19 b_vit_small
python scripts/external_validation.py
python scripts/benchmark_efficiency.py
pip install ultralytics && python scripts/train_yolo_baseline.py --model yolov8n.pt   # optional
python scripts/aggregate_results.py                       # re-run so YOLO joins the detection table
```

| Paper element | Produced by | File |
|---|---|---|
| Table: dataset & split audit | `make_splits.py` | `data/splits/leakage_report.json` |
| Table: main comparison | `aggregate_results.py` | `results/tables/main_comparison.tex` |
| Table: leakage study + figure | `aggregate_results.py` | `results/tables/leakage.tex`, `results/figures/leakage.png` |
| Table: ablations | `aggregate_results.py` | `results/tables/ablation.tex` |
| Table: detection | `aggregate_results.py` | `results/tables/detection.tex` |
| Table: significance | `aggregate_results.py` | `results/tables/significance.tex`, `pooled_oof.csv` |
| Table: efficiency | `benchmark_efficiency.py` | `results/tables/efficiency.tex` |
| Table: external validation | `external_validation.py` | `results/tables/external_validation.csv` |
| Fig: confusion / ROC / reliability / curves | `train.py` | `results/runs/kidneymtl_main/fold*/plots/` |
| Fig: Grad-CAM + pointing game | `explain.py` | `results/explain/` |
| Fig: detection examples | `train.py` | `results/runs/kidneymtl_main/fold*/plots/detections.png` |

### Outputs of one run (`results/runs/<experiment>/fold<k>/`)

`config.yaml` (exact config) · `train.log` · `history.csv` (per-epoch losses/metrics/LR/task weights) ·
`best.pt` (EMA weights, best validation macro-F1) · `{val,test}_metrics.json` (all metrics, per-class, confusion matrix,
stone-vs-rest) · `{val,test}_predictions.csv` (per-image probabilities, used for pooled stats) ·
`det_{val,test}_metrics.json` · `external_metrics.json` (unseen Dataset-1 test) · `summary.json` · `plots/` · `DONE`
(or `error.log` if the run crashed). `best.pt` is kept for all folds of the main model and for fold 0 of the baselines.

## 5. Methodological checklist (what reviewers will look for)

- [x] Split **before** augmentation; augmentation only on the training split, online.
- [x] Near-duplicate-aware grouped folds (no patient IDs exist; pHash groups act as a proxy). This limitation is stated in the paper.
- [x] Model selection on a validation fold that is disjoint from the test fold; test touched once per fold.
- [x] 5-fold CV mean ± std, pooled out-of-fold 95 % cluster-bootstrap CIs, paired McNemar tests with Holm correction.
- [x] Class imbalance handled (focal loss + class weights) and reported with balanced accuracy / macro-F1 / MCC.
- [x] Calibration (ECE, Brier, reliability diagrams), not just accuracy.
- [x] The reference method re-implemented and evaluated under the identical protocol.
- [x] Ablations isolating every component; detection-only and classification-only controls for the MTL claim.
- [x] Quantitative explainability (pointing game) besides qualitative heatmaps.
- [x] External, cross-dataset test (Dataset 1 is coronal, Dataset 2 axial; reported as a limitation).
- [x] Seeds, exact configs, folds and package versions are saved, and the code is public.
- [ ] Fill in the CLAIM checklist (Mongan et al., 2020) as supplementary material.

## Notes

* `kaggle.json` and the datasets are **not** in the repository. The Kaggle API key that used to be in
  `notebooks/03_data_augmentation.ipynb` has been removed, but it was exposed before. **Revoke it at
  kaggle.com → Settings → API and create a new token.**
* The reference PDF is not committed (copyright). Cite it: Devi M. S. et al., "Contour-Detected Normalized Residual
  Model for Kidney Stone Classification", *Applied Computational Intelligence and Soft Computing*, 2025,
  doi:10.1155/acis/5844438.

## Author

**Utkarsh Kumar**
