# EE 6770 – Final Project: Individual Progress Report
**Student:** Olasubomi Rufai
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
  - (If run) `comparison_baselines.png` (Macro-F1 vs Accuracy bar charts).
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

### Overfitting Analysis

| Epoch | Train Loss | Train Macro-F1 | Val Loss | Val Macro-F1 | Notes |
|-------|-----------:|---------------:|---------:|-------------:|-------|
| 1     | 0.2207     | 0.9071         | 0.8263   | **0.5772**    | Best validation; checkpoint saved |
| 2     | 0.1218     | 0.9524         | 1.2085   | 0.4987        | Validation performance drops sharply |
| 3     | 0.1012     | 0.9602         | 1.4190   | 0.4782        | Further decline; overfitting intensifies |

Pattern: Training metrics improve (loss ↓, F1 ↑) while validation loss increases and Macro-F1 decreases after epoch 1. This divergence indicates memorization of training samples rather than generalizable features.

Mitigation steps now in place: backbone freezing, increased dropout, stronger weight decay, lower learning rate, early stopping patience reduction, lighter alternative architecture (EfficientNet-B0), and improved data loading throughput (local copy + fewer workers).

**Challenge: Training Time (Bold)** – In addition to overfitting, the throughput slowdown (≈5.25 s/iteration mid-epoch when reading from Drive with 4 workers) inflated epoch duration, making rapid experimentation impractical until mitigations (local copy, fewer workers) were applied.

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

### Detailed Training Time Breakdown (ResNet50 Baseline)

| Phase / Epoch | Train Duration | Val Duration | Notes |
|---------------|---------------:|-------------:|-------|
| Epoch 1       | ~42 min        | ~32 min      | Includes initial model + caching overhead |
| Epoch 2       | ~5 min         | ~2.3 min     | After caching; faster I/O |
| Epoch 3       | ~5 min (train) | ~2.4 min     | Validation degraded; overfitting intensified |
| Drive-I/O worst case (observed mid-epoch) | >70 min (projected) | >30 min (projected) | When streaming directly from Drive with 4 workers and contention |

**Observation:** After the first epoch warms caches, training becomes much faster if data access is stable. However, reading via Google Drive with high worker count can revert to slow per-iteration times (~5.25s/it). Copying the dataset locally plus reducing workers stabilizes throughput.

### Test Set Results (Best Checkpoint – Epoch 1)

| Metric            | Value |
|-------------------|------:|
| Accuracy          | 0.5818 |
| Macro Precision   | 0.6219 |
| Macro Recall      | 0.6152 |
| Macro F1          | 0.5806 |
| ROC AUC (macro)   | 0.6415 |

Per-Class Detail:

| Class      | Precision | Recall | F1     | Support |
|------------|----------:|-------:|-------:|--------:|
| notdrowsy  | 0.4886    | 0.7877 | 0.6031 |  8,846 |
| drowsy     | 0.7552    | 0.4427 | 0.5581 | 13,087 |

**Interpretation:** Model favors higher recall for notdrowsy and higher precision for drowsy, indicating a conservative drowsy classification (missed positives) under current training regime.

### Figure References

![Confusion Matrix (Test)](../reports/figures/confusion_matrix_test.png)
![Confusion Matrix Normalized (Test)](../reports/figures/confusion_matrix_test_normalized.png)
![ROC Curves (Test)](../reports/figures/roc_curves_test.png)
<!-- Optional baseline comparison if generated -->
<!-- ![Baseline Comparison](../reports/figures/comparison_baselines.png) -->

---
---

## Part 2: Collaboration & Next Steps (30%)

### 1. How We Communicate or Share Progress

Our team coordinates mainly through online messaging and short meetings to align on tasks. We use the shared GitHub repository ([nthu-driver-drowsiness-ROI](https://github.com/hmolhem/nthu-driver-drowsiness-ROI)) as our central place to push code, track commits, and review changes. When needed, we also share experiment logs and notebooks (e.g., from Colab) so that everyone can see the latest results and reproduce runs on their own machines.

Hossein Molhem coordinates the project and handles core implementation. My role focuses on presentation preparation and results communication.

### 2. What I Will Personally Complete Before the Final Presentation

**Assigned Tasks (Presentation & Visualization):**

- **Week 1 (Nov 19–24):**
  - Design presentation slide structure: Introduction, Problem Statement, Methodology, Results, Conclusion
  - Create visualizations comparing baseline models (ResNet50, Regularized ResNet50, EfficientNet-B0)
  - Develop infographics explaining ROI-gated approach and architecture diagrams
  - Generate performance comparison charts (accuracy, F1-score, ROC curves across models)
  - Design training progress visualizations (loss curves, overfitting analysis, time consumption breakdown)
  - Prepare dataset overview slides with subject-exclusive split explanation

- **Week 2 (Nov 25–Dec 1):**
  - Integrate ROI model results from Tasfia into presentation slides
  - Create final comparison tables: Baseline vs. ROI performance metrics
  - Develop demonstration materials: sample predictions, attention visualizations, confusion matrices
  - Prepare technical slides explaining challenges (overfitting, training time) and solutions
  - Assemble final presentation deck with consistent formatting and clear narrative flow
  - Coordinate presentation rehearsal with team and refine based on feedback
  - Prepare speaker notes and timing for each section

- **Final Report Assembly:**
  - Consolidate technical documentation from all team members
  - Format final project report with sections: Abstract, Introduction, Methodology, Experiments, Results, Conclusion
  - Integrate figures, tables, and code snippets from repository
  - Proofread and ensure consistent terminology and formatting
  - Generate PDF version for submission

- **Final Deliverables:**
  - Complete presentation slides (PowerPoint/Google Slides)
  - Final project report (PDF)
  - Supporting visualization materials
  - Presentation script and timing plan

### 3. Coordination or Workload Challenges

**Team Coordination:**
- Timing dependency: Presentation slides require completed experimental results from Hossein and Tasfia
- Ensuring technical accuracy in visualizations and explanations while making content accessible
- Coordinating presentation rehearsal timing across team schedules

**Presentation Challenges:**
- Balancing technical depth with clarity: explaining ROI approach without overwhelming audience
- Time constraint: fitting comprehensive project overview into presentation time limit
- Visual design: creating professional, publication-quality figures from raw experimental outputs

**Mitigation:**
- Early slide template creation to parallelize content development
- Regular check-ins with Hossein and Tasfia to stay updated on latest results
- Iterative slide review process with team feedback before final version
- Preparing backup slides with additional details in case of questions

**Current Status:**
Work is on track. Baseline results are available for initial slide development. ROI results expected by end of Week 1, allowing Week 2 for integration and refinement.

---

## Export to PDF (Options)

- VS Code: Open this file and use "Markdown: Print to PDF" (or the Markdown PDF extension).
- Pandoc (if installed):
```powershell
pandoc docs/individual-progress-report.md -o docs/individual-progress-report.pdf
```

---

## Repository Links

- **GitHub Repository:** [hmolhem/nthu-driver-drowsiness-ROI](https://github.com/hmolhem/nthu-driver-drowsiness-ROI)
- **Documentation Folder:** [docs/](https://github.com/hmolhem/nthu-driver-drowsiness-ROI/tree/main/docs)
- **Evaluation Figures:** [reports/figures/](https://github.com/hmolhem/nthu-driver-drowsiness-ROI/tree/main/reports/figures)
- **Training Notebooks:** [notebooks/colab_gpu_training.ipynb](https://github.com/hmolhem/nthu-driver-drowsiness-ROI/blob/main/notebooks/colab_gpu_training.ipynb)
- **Latest Release:** [v0.1-eval-baseline](https://github.com/hmolhem/nthu-driver-drowsiness-ROI/releases/tag/v0.1-eval-baseline)
