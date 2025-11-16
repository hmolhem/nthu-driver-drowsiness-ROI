# Driver Drowsiness Detection Project Proposal

**Course:** EE6770 Applications of Neural Networks (Fall 2025)  
**Team:** Hossein Molhem · Olasubomi Rufai · Tasfia Kabir  
**Faculty:** Dr. Jeffrey L. Yiin

---
## 1. Problem Statement
A significant road safety concern is drowsy or fatigued driving. We build a **single-image** (no temporal modeling) TensorFlow/Keras pipeline on a Kaggle mirror of the **NTHU–DDD2** dataset to:
1. **Segmentation (auxiliary):** Estimate region-of-interest (ROI) masks for eyes and mouth to focus on blink/yawn cues.
2. **Classification (primary):** Predict driver state (normal / slow_blink / yawn / sleepy) from each image.

Segmentation is used as a hypothesis-driven prior to improve robustness in a constrained scenario (day + glasses). No recurrent/video temporal models are included.

---
## 2. Objectives
- Clean, reproducible per-image pipeline: ingestion → preprocessing → pseudo-mask generation → optional segmentation → classification.
- Evaluate compact CNN backbones (EfficientNetV2-S, ResNet50, VGG19); compare plain vs. ROI-assisted vs. multi-task shared encoder.
- Report robustness only for available scenario (day+glasses) using synthetic stress tests (glare, eyelid occlusion, mild blur).
- Enforce subject-exclusive splits; audit identity leakage impact on macro-F1.
- Quantify compactness vs. accuracy trade-off (parameters/FLOPs vs. macro-F1).

---
## 3. Novelty & Contribution
| Typical Prior Work | Our Added Value |
|--------------------|-----------------|
| Random frame splits risk identity leakage | Subject-exclusive splits + leakage audit |
| Image-only CNN accuracy focus | Explicit robustness slices (synthetic glare/occlusion/blur) |
| Heavy backbone benchmarks | Compact backbone + ROI prior efficiency comparison |

Segmentation itself is **not** the headline; evaluation rigor and trustworthy metrics are.

---
## 4. Dataset Subset Facts
| Item | Value |
|------|-------|
| Total images | **66,532** (subset) |
| Subjects | **4** |
| Class counts | normal 30,590 · slow_blink 9,413 · yawn 8,863 · sleepy 17,666 |
| Scenario coverage | day-glasses only |
| Max/min class ratio | 3.45 |

All splits are **subject-exclusive**. If subject ambiguity arises, fallback to stratified 5-fold ensuring subject frames do not cross folds.

---
## 5. Preprocessing
- Resize to 224×224 (alternate 160 for ablation).
- Per-image normalization; optional CLAHE for low-light if added later.
- Corrupt frame removal + label consistency checks.

---
## 6. Pseudo-Mask Segmentation (Auxiliary)
- Generate eye & mouth binary masks via classical detection (OpenCV Haar cascade / landmarks).
- Manual QC subset with Dice; masks failing quality gate degrade segmentation strategy priority.
- U-Net encoder–decoder (Dice + BCE loss).

---
## 7. Classification Model
- Backbones: EfficientNetV2-S, ResNet50, VGG19 (ImageNet init).
- Head: Global pooling → Dense classifier (label smoothing). Loss: Categorical CE (Focal optional for imbalance).
- Optimizer: AdamW + cosine decay + warmup; early stopping on validation macro-F1.
- Augmentations: horizontal flip, mild color jitter, light blur; conservative to avoid semantic distortion.

---
## 8. Segmentation → Classification Interfaces
Let image \(x \in \mathbb{R}^{H\times W\times 3}\) and predicted mask \(M \in [0,1]^{H\times W}\).
1. **Pre-mask gating:** \(\tilde{x} = x \odot \text{expand}(M)\)
2. **Multi-task shared encoder:** Encoder \(E(x)\) branches to segmentation \(D(E(x))→M\) and classification \(C(E(x))→\hat{y}\). Joint loss:
\[ \mathcal{L} = \lambda_{seg}(Dice + BCE) + \lambda_{cls} CE. \]

---
## 9. Ablations
- Backbone: ResNet50 vs. EfficientNetV2-S vs. VGG19
- Input size: 160 vs. 224
- Augmentation strength: base vs. stronger color/light
- Mask strategy: none vs. mask×image vs. mask-as-channel vs. multi-task

---
## 10. Metrics & Evaluation
- **Primary:** Macro-Precision, Macro-Recall, Macro-F1 (with confusion matrix).
- **Secondary:** Accuracy, per-class F1, optional ROC-AUC.
- **Segmentation:** Dice, IoU on QC pseudo-mask subset.
- **Baselines:** Majority-class predictor; Plain CNN; ROI pre-mask gating.
- **Robustness:** Performance under synthetic glare / eyelid occlusion / mild blur (only for day+glasses).

---
## 11. Quality Gates
| Gate | Threshold / Action |
|------|--------------------|
| Data counts | All splits sum to 100%; subject-exclusive verified |
| Pseudo-mask quality | Dice < 0.75 (QC subset) → fallback to no-mask baseline in final report (still document results) |
| Baseline performance | Plain CNN macro-F1 ≥ 0.80 |
| Final target | Test macro-F1 ≥ 0.88 (stretch ≥ 0.92 aspirational) |

---
## 12. Timeline (4 Weeks)
| Week | Focus |
|------|-------|
| 1 | Data ingestion, manifest, pseudo-mask v1, baseline CNN |
| 2 | ROI gating variants, multi-task start, metrics stabilization |
| 3 | Robustness transforms, ablations, QC overlays |
| 4 | Final training, tables/figures, polishing report & reproducibility |

---
## 13. Responsibilities (Condensed RACI)
| Item | Responsible | Support | Notes |
|------|------------|---------|-------|
| Data manifest & counts | Hossein | Tasfia | Build `manifest.csv` |
| Subject-aware splits | Tasfia | Hossein | 70/15/15 or stratified folds |
| Pseudo-masks (OpenCV) | Hossein | Olasubomi | v1/v2 tuning |
| U-Net training | Hossein | — | Dice+BCE |
| Baseline CNN | Olasubomi | — | Early stop macro-F1 |
| Pre-mask gating | Olasubomi | Hossein | multiply/concat |
| Multi-task encoder | Hossein | Olasubomi | joint loss monitor |
| Metrics & plots | Tasfia | — | confusion, slices |
| Ablations | Olasubomi | Tasfia | backbone/size/aug/mask |
| Paper figs/tables | Tasfia | Hossein | counts, curves |
| Repro pack | Hossein | All | one-command reproduce |

---
## 14. Reproducibility Practices
- Fixed seeds & deterministic data shuffling where feasible.
- YAML configs per experiment; logged hyperparameters.
- Versioned environment (`requirements.txt` + Python 3.10).
- Store run artifacts: metrics JSON, confusion matrices, segmentation QC.

---
## 15. Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|-----------|
| Only 4 subjects | High variance / limited generalization | Document constraint; consider adding more subjects if accessible |
| Mask generation noise | Degrades ROI benefit | QC subset + fallback rule |
| Scenario limitation (day+glasses) | Robustness claims narrow | Explicit scenario labeling; avoid over-generalization |
| Class imbalance | Biased metrics | Macro-F1 focus; optional class weights/focal loss |

---
## 16. Immediate Next Steps (Actionable)
1. Implement `data/manifests/build_manifest.py` to produce `manifest_master.csv` with columns: `filepath, subject_id, class_label, scenario, split`.
2. Add leakage audit script: `src/eval/leakage_audit.py` (random split vs. subject-exclusive macro-F1 comparison).
3. Scaffold pseudo-mask generation (`src/masks/build_pseudo_masks.py`) with OpenCV eye/mouth detection; save masks to `masks_eyes/` and `masks_mouth/`.
4. Add Dice + BCE segmentation loss utilities to `src/training/losses.py` (if not present).
5. Create config templates: `configs/baseline_resnet50.yaml`, `configs/roi_efficientnet.yaml`, `configs/multitask_unet_classifier.yaml`.
6. Extend `src/training/metrics.py` for macro metrics + Dice/IoU; ensure consistent naming.
7. Implement ROI gating functions in `src/models/roi_gating.py` (`apply_mask(image, mask, mode)`; supports `multiply`, `concat`).
8. Add robustness transforms in `src/data/transforms.py` (`add_glare`, `eyelid_occlusion`, `mild_blur`) with seed control.
9. Logging enhancements: Include `mask_strategy`, `backbone`, `input_size`, `aug_profile`, `multitask` tags.
10. Prepare `docs/` scaffolds: `dataset.md`, `masks.md`, `models.md`, `training.md`, `robustness.md`, `ablations.md`, `quality_gates.md`, `reproduce.md`.

---
## 17. Stretch Goals
- MobileNetV3-Small baseline for efficiency comparison.
- FLOPs/parameter auto-computation script.
- Deployment-friendly ONNX export for the best model.

---
## 18. References
1. NTHU–DDD2 Kaggle Mirror: https://www.kaggle.com/datasets/banudeep/nthuddd2/data
2. Ronneberger et al. U-Net (MICCAI 2015)
3. Tan & Le, EfficientNetV2 (ICML 2021)
4. He et al., ResNet (CVPR 2016)
5. Simonyan & Zisserman, VGG (ICLR 2015)

---
**Status:** Proposal captured; awaiting confirmation to proceed with implementation steps (manifest + mask pipeline).
