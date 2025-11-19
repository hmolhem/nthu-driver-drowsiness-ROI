# EE 6770 – Final Project: Individual Progress Report
**Student:** [Your Name]
**Project:** NTHU Driver Drowsiness Detection with ROI
**Date:** November 19, 2025

---

## Part 1: Technical Contribution & Learning (70%)

- **Main Roles:**
  - Data prep validation (subject-exclusive splits under `data/splits/`).
  - Baseline modeling (ResNet50) and training orchestration in Colab.
  - Evaluation tooling (metrics/plots) and experiment reporting.
  - Runtime analysis and mitigation for time-consuming Colab training.

- **Implemented & Tested:**
  - `src/eval/evaluate_model.py` (new CLI):
    - Computes accuracy, macro precision/recall/F1, per-class metrics.
    - Saves confusion matrix CSV and PNGs (raw + normalized), ROC curves (per class) with AUC.
    - Exports `predictions_test.csv` with probabilities per class.
  - Colab workflow (`notebooks/colab_gpu_training.ipynb`) upgrades:
    - Regularized ResNet50 config writer (freeze backbone, dropout 0.7, LR 5e-5, WD 5e-4, patience 5, workers 2).
    - Train/eval/export cells for Regularized ResNet50 and EfficientNet-B0.
    - Comparison cell to plot Accuracy and Macro-F1 across baselines.
    - Speed-up & resume cells: copy dataset to VM SSD, patch `num_workers: 2`.
  - Reports:
    - `docs/training-report-nov18-colab.md`: training status, overfitting analysis, timings, test results, next steps.
    - `docs/report-2025-11-18.md`: daily summary and decision to stop training (time + generalization).

- **Evidence (Code, Models, Figures):**
  - Checkpoints: `checkpoints/baseline_resnet50_best.pth` (epoch 1), `baseline_resnet50_last.pth`.
  - Test (ResNet50 best): Accuracy 0.5818; Macro-F1 0.5806; ROC AUC (macro) 0.6415.
  - Generated figures (copied to `reports/figures/`):
    - `confusion_matrix_test.png`, `confusion_matrix_test_normalized.png`, `roc_curves_test.png`.
  - Minimal code snippet (evaluation entry):
```
# src/eval/evaluate_model.py (excerpt)
metrics, cls_report, probs, labels = evaluate(model, test_loader, device=device, save_preds_path=preds_csv)
# Save metrics JSON + AUC
metrics_to_save = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in metrics.items()}
metrics_to_save["roc_auc"] = roc_auc_scores
```

- **Challenge & Resolution:**
  - Challenge: Training on Colab became time-consuming with dataset streamed from Google Drive and `num_workers=4` (~5.25 s/iteration mid-epoch), while validation degraded after epoch 1 (classic overfitting).
  - Resolution: Copy dataset to local VM path (`datasets/archive`), reduce `num_workers` to 2, keep `pin_memory: true`. Adopt stronger regularization (freeze backbone, higher dropout, lower LR, higher WD) and add a lighter baseline (EfficientNet-B0) for faster epochs and better generalization.

---

## Part 2: Plan to Complete Before Final Presentation

- Near-term experiments (this week):
  - Train Regularized ResNet50 (Colab, local dataset copy, 2 workers); evaluate and export artifacts.
  - Train EfficientNet-B0; evaluate; run comparison plots (`reports/figures/comparison_baselines.png`).
- ROI phase: Train ROI-gated approach (`src/models/roi_gating.py`, `unet_segmentation.py`) focusing on eyes/mouth.
- Reporting: Consolidate metrics/plots, per-class breakdowns; finalize slides.
- Practical Colab steps:
  - Copy dataset to `/content/nthu-driver-drowsiness-ROI/datasets/archive`, patch `num_workers: 2`, run `--device cuda`.
- Milestones:
  - Week 1 (Nov 19–24): Complete regularized + EfficientNet runs and comparisons.
  - Week 2: ROI training/evaluation; finalize analysis and slides.

---

## Project Snapshot

- Key configs: `configs/baseline_resnet50.yaml`, `configs/baseline_resnet50_regularized.yaml`, `configs/baseline_efficientnet.yaml`.
- Relevant code: `src/training/train_baseline.py`, `src/training/trainer.py`, `src/data/dataset.py`, `src/data/transforms.py`, `src/eval/evaluate_model.py`.
- Notebook: `notebooks/colab_gpu_training.ipynb` (GPU setup → train → evaluate → export → compare → speed-up/resume).
- Figures: under `reports/figures/` (confusion matrices, ROC, comparison).

---

## Colab Time Consumption (GPU) – Findings

- Baseline reference: ~74 minutes for epoch 1 (train ~42m + val ~32m).
- With Drive I/O + 4 workers: ~5.25 s/iteration mid-epoch; epochs elongated and inconsistent.
- Mitigations: use 2 workers, local dataset copy, regularized setup, or EfficientNet-B0.

---

## Export to PDF (Options)

- VS Code: Open this file and use “Markdown: Print to PDF” (or the Markdown PDF extension).
- Pandoc (if installed):
```powershell
pandoc docs/individual-progress-report.md -o docs/individual-progress-report.pdf
```
