"""Registry of every experiment reported in the paper.

Each entry = name -> (group, overrides, paper label). The runner trains
them; the aggregator builds the tables from the same registry, so the paper
tables and the code can never drift apart.
"""
from __future__ import annotations

MAIN = "kidneymtl_main"

_VGG_LR = ["train.lr=1.0e-4", "train.backbone_lr=5.0e-5"]
_CDR = ["model.name=cdr_vgg19", "model.det_enabled=false", "data.input_mode=contour",
        "data.cls_img_size=224", "loss.cls=ce", "loss.class_weighting=none"] + _VGG_LR
_STL_BASE = ["model.name=timm_classifier", "data.input_mode=gray", "loss.cls=ce",
             "loss.class_weighting=none", "model.det_enabled=false"]

EXPERIMENTS: dict[str, dict] = {
    # ---------------------------------------------------------------- main
    MAIN: {"group": "main", "opts": [], "label": "KidneyMTL (ours)"},

    # ------------------------------------------- single-task baselines (STL)
    # standard practice: ImageNet fine-tuning, cross-entropy, grayscale input
    "b_resnet50": {"group": "baselines", "opts": _STL_BASE + ["model.backbone=resnet50"], "label": "ResNet-50"},
    "b_densenet121": {"group": "baselines", "opts": _STL_BASE + ["model.backbone=densenet121"],
                      "label": "DenseNet-121"},
    "b_efficientnet_b3": {"group": "baselines", "opts": _STL_BASE + ["model.backbone=efficientnet_b3"],
                          "label": "EfficientNet-B3"},
    "b_convnext_tiny": {"group": "baselines", "opts": _STL_BASE + ["model.backbone=convnext_tiny"],
                        "label": "ConvNeXt-T"},
    "b_vit_small": {"group": "baselines",
                    "opts": _STL_BASE + ["model.backbone=vit_small_patch16_224", "train.backbone_lr=5.0e-5"],
                    "label": "ViT-S/16"},
    "b_vgg19": {"group": "baselines", "opts": _STL_BASE + ["model.backbone=vgg19_bn"] + _VGG_LR,
                "label": "VGG-19 (BN)"},
    "b_cdr_vgg19": {"group": "baselines",
                    "opts": _CDR,
                    "label": "CDR-VGG19 [ref] (re-impl.)"},

    # ------------------------------------- ablations (one change from MAIN)
    "a_no_det": {"group": "ablations", "opts": ["model.det_enabled=false"], "label": "w/o detection task (STL)"},
    "a_no_attention": {"group": "ablations", "opts": ["model.attention=none"], "label": "w/o CBAM"},
    "a_se": {"group": "ablations", "opts": ["model.attention=se"], "label": "CBAM -> SE"},
    "a_input_gray": {"group": "ablations", "opts": ["data.input_mode=gray"], "label": "input: grayscale"},
    "a_input_clahe": {"group": "ablations", "opts": ["data.input_mode=clahe"], "label": "input: CLAHE"},
    "a_input_contour": {"group": "ablations", "opts": ["data.input_mode=contour"],
                        "label": "input: contour only (ref.-style)"},
    "a_loss_ce": {"group": "ablations", "opts": ["loss.cls=ce", "loss.class_weighting=none"],
                  "label": "focal -> cross-entropy"},
    "a_mtl_fixed": {"group": "ablations", "opts": ["loss.mtl_weighting=fixed"],
                    "label": "uncertainty -> fixed task weights"},
    "a_box_giou": {"group": "ablations", "opts": ["loss.det_box_loss=giou"], "label": "CIoU -> GIoU"},
    "a_act_relu": {"group": "ablations", "opts": ["model.head_activation=relu"], "label": "head act. SiLU -> ReLU"},
    "a_bb_resnet50": {"group": "ablations", "opts": ["model.backbone=resnet50"], "label": "backbone ResNet-50"},
    "a_bb_convnext_tiny": {"group": "ablations", "opts": ["model.backbone=convnext_tiny"],
                           "label": "backbone ConvNeXt-T"},

    # ------------------------------------------- detection-only reference
    "d_det_only": {"group": "detection",
                   "opts": ["train.det_only=true", "train.monitor=val_det_map50", "loss.mtl_weighting=fixed"],
                   "label": "detection only (no cls task)", "folds": [0]},

    # ------------------------------------------------- leakage study
    "l_random_kidneymtl": {"group": "leakage", "opts": ["data.split_protocol=random"],
                           "label": "KidneyMTL, random split"},
    "l_augsplit_kidneymtl": {"group": "leakage", "opts": ["data.split_protocol=aug_then_split"],
                             "label": "KidneyMTL, augment-then-split", "folds": [0, 1, 2]},
    "l_random_resnet50": {"group": "leakage",
                          "opts": _STL_BASE + ["model.backbone=resnet50", "data.split_protocol=random"],
                          "label": "ResNet-50, random split"},
    "l_augsplit_resnet50": {"group": "leakage",
                            "opts": _STL_BASE + ["model.backbone=resnet50", "data.split_protocol=aug_then_split"],
                            "label": "ResNet-50, augment-then-split", "folds": [0, 1, 2]},
    "l_random_cdr_vgg19": {"group": "leakage",
                           "opts": _CDR + ["data.split_protocol=random"],
                           "label": "CDR-VGG19, random split"},
    "l_augsplit_cdr_vgg19": {"group": "leakage",
                             "opts": _CDR + ["data.split_protocol=aug_then_split"],
                             "label": "CDR-VGG19, augment-then-split (ref. protocol)", "folds": [0, 1, 2]},
}

GROUP_ORDER = ["main", "baselines", "ablations", "detection", "leakage"]
ALL_FOLDS = [0, 1, 2, 3, 4]


ABLATION_FOLDS = [0, 1, 2]

# Run order if compute is limited: tier 1 = minimum publishable set.
TIER1 = [MAIN, "b_cdr_vgg19", "b_resnet50", "b_efficientnet_b3", "b_convnext_tiny", "a_no_det",
         "a_input_gray", "l_random_kidneymtl", "l_augsplit_cdr_vgg19", "l_random_cdr_vgg19"]


def folds_for(name: str) -> list[int]:
    default = ABLATION_FOLDS if EXPERIMENTS[name]["group"] == "ablations" else ALL_FOLDS
    return EXPERIMENTS[name].get("folds", default)


def experiment_opts(name: str) -> list[str]:
    opts = list(EXPERIMENTS[name]["opts"]) + [f"experiment.name={name}"]
    if EXPERIMENTS[name]["group"] == "ablations":
        opts.append("experiment.keep_checkpoint=false")
    return opts


def protocol_of(name: str) -> str:
    for o in EXPERIMENTS[name]["opts"]:
        if o.startswith("data.split_protocol="):
            return o.split("=", 1)[1]
    return "grouped"
