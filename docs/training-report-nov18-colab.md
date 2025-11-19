# Training Report - Baseline ResNet50
**Date:** November 18, 2025  
**Environment:** Google Colab (GPU: Tesla T4/L4, CUDA 12.1)  
**Model:** ResNet50 (pretrained on ImageNet)  
**Task:** Driver Drowsiness Detection (Binary Classification)

---

## Executive Summary

Successfully initiated baseline ResNet50 training on GPU with subject-exclusive data splits. Completed 3 epochs before observing severe overfitting behavior. **Best model achieved 57.72% validation Macro-F1 on epoch 1**, which is a reasonable baseline but indicates the model struggles to generalize from the training data. The saved best checkpoint (epoch 1) evaluated on the test set achieved **58.18% accuracy**, **0.5806 Macro-F1**, and **0.6415 ROC AUC (macro)**.

**Key Finding:** Model exhibits classic overfitting pattern - high training accuracy (95%+) but declining validation performance (F1 dropping from 0.577 → 0.499). Recommending early termination and model adjustments before continuing.

---

## Dataset Configuration

### Data Splits (Subject-Exclusive)
- **Training Set:** 25,572 samples
  - Drowsy: 13,359 (52.2%)
  - Not Drowsy: 12,213 (47.8%)
  
- **Validation Set:** 19,016 samples
  - Drowsy: 9,584 (50.4%)
  - Not Drowsy: 9,432 (49.6%)
  
- **Test Set:** 21,933 samples
  - Drowsy: 13,087 (59.7%)
  - Not Drowsy: 8,846 (40.3%)

### Data Splits Source
- Train: `data/splits/train.csv`
- Validation: `data/splits/val.csv`
- Test: `data/splits/test.csv`

### Preprocessing & Augmentation
- **Image Size:** 224×224 pixels
- **Normalization:** ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Training Augmentation:**
  - Horizontal flip (p=0.5)
  - Color jitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
  - Random rotation (±10°)
  - Random affine (translate=0.1, scale=0.9-1.1)

---

## Model Architecture

### Backbone
- **Architecture:** ResNet50 (pretrained on ImageNet1K)
- **Total Parameters:** 23,512,130
- **Trainable Parameters:** 23,512,130 (all layers fine-tuned)
- **Dropout:** 0.5 (before final classifier)
- **Output:** 2 classes (notdrowsy, drowsy)

### Training Configuration
- **Optimizer:** Adam
- **Learning Rate:** 1e-4 (0.0001)
- **Weight Decay:** 1e-4
- **Batch Size:** 32
- **Loss Function:** Weighted Cross-Entropy (class weights applied)
- **LR Scheduler:** ReduceLROnPlateau
  - Mode: max (monitor validation F1)
  - Factor: 0.5
  - Patience: 5 epochs
  - Min LR: 1e-6

### Early Stopping
- **Enabled:** Yes
- **Monitor Metric:** Validation Macro-F1
- **Patience:** 10 epochs
- **Mode:** Maximize F1

---

## Training Results

### Epoch 1 (Completed)
**Training Phase:** 42 minutes 1 second
- Loss: 0.2207
- Accuracy: 90.74%
- Macro-F1: 0.9071
- Speed: ~3.16 sec/batch

**Validation Phase:** 31 minutes 45 seconds
- Loss: 0.8263
- Accuracy: **58.12%**
- Macro-F1: **0.5772** ✅ **BEST MODEL SAVED**
- Speed: ~3.20 sec/batch

**Status:** ✅ Best checkpoint saved to `checkpoints/baseline_resnet50_best.pth`

---

### Epoch 2 (Completed)
**Training Phase:** 5 minutes 6 seconds
- Loss: 0.1218 (↓ 45% from epoch 1)
- Accuracy: 95.25% (↑ 4.5%)
- Macro-F1: 0.9524 (↑ 0.045)
- Speed: ~2.61 it/sec (much faster than epoch 1)

**Validation Phase:** 2 minutes 23 seconds
- Loss: 1.2085 (↑ 46% from epoch 1) ⚠️
- Accuracy: **53.87%** (↓ 4.3%) ⚠️
- Macro-F1: **0.4987** (↓ 0.08) ⚠️
- Speed: ~4.14 it/sec

**Status:** ❌ Performance degraded - no checkpoint saved

---

### Epoch 3 (Completed)
**Training Phase:** 5 minutes 8 seconds
- Loss: 0.1012 (↓ 17% from epoch 2)
- Accuracy: 96.03%
- Macro-F1: 0.9602
- Speed: ~2.59 it/sec

**Validation Phase:** 2 minutes 22 seconds
- Loss: 1.4190 (↑ vs epoch 2) ⚠️
- Accuracy: **47.90%** (↓ vs epoch 2) ⚠️
- Macro-F1: **0.4782** (↓ vs epoch 2) ⚠️

**Status:** ❌ Performance further degraded - overfitting intensifying

### Epoch 4 (Interrupted)
**Training Phase:** 93% complete (747/799 batches)
- Loss: 0.0880
- Learning rate: 1e-4

Run was interrupted by user KeyboardInterrupt during training:
```
KeyboardInterrupt at trainer.train_epoch() → metrics_calc.update()
```

**Status:** ⏹ Stopped intentionally to avoid wasting GPU time

---

## Performance Analysis

### Training vs Validation Gap

| Metric | Train (Epoch 3) | Val (Epoch 2) | Gap |
|--------|----------------|---------------|-----|
| Accuracy | 96%+ | 53.87% | **42%+** |
| Macro-F1 | 0.96+ | 0.4987 | **0.46+** |
| Loss | 0.1012 | 1.2085 | **10.9×** |

### Trend Analysis

**Training Metrics (Improving):**
- Loss: 0.2207 → 0.1218 → 0.1012 (steady decrease ✓)
- Accuracy: 90.74% → 95.25% → 96%+ (steady increase ✓)
- Macro-F1: 0.9071 → 0.9524 → 0.96+ (steady increase ✓)

**Validation Metrics (Degrading):**
- Loss: 0.8263 → 1.2085 → 1.82+ (steady **increase** ✗)
- Accuracy: 58.12% → 53.87% → ? (steady **decrease** ✗)
- Macro-F1: 0.5772 → 0.4987 → ? (steady **decrease** ✗)

**Interpretation:** Classic overfitting pattern - model is memorizing training data rather than learning generalizable features.

---

## Critical Issues Identified

### 1. Severe Overfitting ⚠️
**Evidence:**
- Training accuracy (96%+) vs validation accuracy (54%) = **42% gap**
- Validation loss increasing while training loss decreasing
- Validation F1 score dropped 13.6% from epoch 1 to epoch 2

**Possible Causes:**
- Model complexity too high (23.5M parameters for 25K samples)
- Insufficient regularization (dropout=0.5 may be too low)
- Data augmentation not aggressive enough
- Potential data leakage in splits (needs verification)
- Subject-exclusive splits may create domain shift

### 2. DataLoader Workers
- Colab warned: suggested max workers 2; current config used 4.
- While not the root cause, mismatched workers can slow or destabilize throughput; set `num_workers: 2` for Colab.

### 2. Class Imbalance (Test Set)
- Test set: 59.7% drowsy vs 40.3% not drowsy (19.4% imbalance)
- Training/val sets are more balanced (~50/50)
- May affect final test performance evaluation

### 3. Training Speed Inconsistency
- Epoch 1: 42 minutes (initial model download + warmup)
- Epochs 2-3: ~5 minutes (normal speed)
- Validation speed variable (2-32 minutes) - likely DataLoader worker warning effect

---

## Saved Artifacts

### Checkpoints
✅ **Best Model:** `checkpoints/baseline_resnet50_best.pth`
- Epoch: 1
- Validation Macro-F1: 0.5772
- Validation Accuracy: 58.12%
- File size: ~90 MB

✅ **Last Model:** `checkpoints/baseline_resnet50_last.pth`
- Epoch: 2 (last completed)
- File size: ~90 MB

### Configuration
- Config file: `configs/baseline_resnet50.yaml`
- Training script: `src/training/train_baseline.py`
- Random seed: 42

---

## Test Evaluation (Best Checkpoint)

Evaluated the saved best checkpoint (`epoch 1`) on the subject-exclusive test set using the evaluation CLI.

- **Accuracy:** 58.18%
- **Macro F1:** 0.5806
- **ROC AUC (macro):** 0.6415

Per-class metrics (from `sklearn` report):
- notdrowsy — precision: 0.4886, recall: 0.7877, f1: 0.6031, support: 8,846
- drowsy — precision: 0.7552, recall: 0.4427, f1: 0.5581, support: 13,087

Figures (saved during evaluation):
- Confusion Matrix (raw counts): `runs/baseline_resnet50/confusion_matrix_test.png`
- Confusion Matrix (normalized): `runs/baseline_resnet50/confusion_matrix_test_normalized.png`
- ROC Curves (per-class + chance): `runs/baseline_resnet50/roc_curves_test.png`

Copies are also available under `reports/figures/` for convenient access in reports and slides.

Preview (local render-only; images are generated artifacts and may be git-ignored):

![Confusion Matrix (Test)](../reports/figures/confusion_matrix_test.png)

![Confusion Matrix (Normalized)](../reports/figures/confusion_matrix_test_normalized.png)

![ROC Curves (Test)](../reports/figures/roc_curves_test.png)

---

## Recommendations for Next Steps

### Immediate Actions (Before Professor Meeting)

1. **Stop Training Now** ✅
   - Current best model (epoch 1) is already saved
   - Continuing wastes GPU time with degrading results
   - Can demonstrate understanding of overfitting

2. **Evaluate Best Checkpoint on Test Set**
   ```python
   # Run in Colab after stopping training
   !python src/eval/evaluate_model.py \
     --config configs/baseline_resnet50.yaml \
     --device cuda \
     --save-fig \
     --save-preds
   ```
   - Get final test metrics, confusion matrix, ROC curves
   - Download results for professor presentation

3. **Download All Results**
   ```python
   # In Colab
   !zip -r results.zip checkpoints/ runs/baseline_resnet50/
   from google.colab import files
   files.download('results.zip')
   ```

### Model Improvements for Next Training Run

**Strategy A: Stronger Regularization (Applied in new config)**
- Increase dropout: 0.5 → 0.7 ✅
- Freeze backbone initially: `freeze_backbone: true` ✅
- Reduce LR: 1e-4 → 5e-5 ✅
- Increase weight decay: 1e-4 → 5e-4 ✅
- Early stopping patience: 10 → 5 ✅
- DataLoader workers: 4 → 2 (Colab-friendly) ✅
- Note: augmentation left as-is (code uses fixed transforms); can extend later.

**Strategy B: Simpler Architecture**
- Try EfficientNet-B0 (~5M parameters vs 23M)
- Freeze backbone, train only classifier head
- Reduce model capacity to match dataset size

**Strategy C: Data Verification**
- Verify subject-exclusive splits are correct
- Check for potential data leakage
- Analyze per-subject performance distribution
- Consider k-fold cross-validation

**Strategy D: Learning Rate Adjustment**
- Reduce initial LR: 1e-4 → 1e-5
- Use warmup schedule
- Try cosine annealing instead of ReduceLROnPlateau

---

## Professor Presentation Summary

### What to Report

**Achievements:**
✅ Successfully set up PyTorch training pipeline  
✅ Implemented subject-exclusive data splits  
✅ Trained baseline ResNet50 on GPU (Colab)  
✅ Completed 3 epochs with proper monitoring  
✅ Saved best checkpoint (val F1: 0.5772)  
✅ Identified overfitting early through validation metrics  

**Challenges:**
⚠️ Model shows severe overfitting (42% train-val gap)  
⚠️ Validation performance degraded after epoch 1  
⚠️ Baseline F1 of 0.577 indicates difficult task  

**Next Steps:**
🎯 Evaluate best model on test set  
🎯 Implement stronger regularization  
🎯 Try lighter architecture (EfficientNet-B0)  
🎯 Verify data splits for leakage  
🎯 Compare with ROI-gated approach (project goal)  

---

## Technical Specifications

### Hardware
- **Platform:** Google Colab
- **GPU:** Tesla T4 or L4
- **CUDA Version:** 12.1
- **Python:** 3.12
- **PyTorch:** 2.5.1+cu121

### Software Environment
- Framework: PyTorch
- Key Libraries: torchvision, scikit-learn, pandas, matplotlib, seaborn
- Version Control: Git (repository: nthu-driver-drowsiness-ROI)

### Training Time
- **Total elapsed:** ~1 hour 35 minutes (3 partial epochs)
- **Epoch 1:** 74 minutes (train + val)
- **Epoch 2:** 7.5 minutes (train + val)
- **Epoch 3:** 5 minutes (train only, incomplete)
- **Estimated full run:** 10-30 hours (if continued to early stopping)

---

## Conclusion

Successfully established baseline drowsiness detection model with proper experimental setup. The model achieves **57.72% validation Macro-F1** but exhibits severe overfitting, indicating the need for regularization improvements before final evaluation. Best checkpoint is saved and ready for test set evaluation.

**Recommended Approach:** Stop training now, evaluate epoch 1 checkpoint on test set for baseline metrics, then restart with adjusted hyperparameters for improved generalization.

---

## Appendix: How to Resume/Stop Training

### Stop Training in Colab
```
Press: Ctrl + M, then I (keyboard interrupt)
Or: Click the stop button in Colab
```

### Evaluate Current Best Model
```bash
# In Colab (after stopping)
!python src/eval/evaluate_model.py \
  --config configs/baseline_resnet50.yaml \
  --checkpoint checkpoints/baseline_resnet50_best.pth \
  --device cuda \
  --save-fig \
  --save-preds
```

### Train and Evaluate Regularized Baseline (Overfitting Mitigation)
```bash
# Write regularized config (already included in notebook automation)
# Train
!python src/training/train_baseline.py \
  --config configs/baseline_resnet50_regularized.yaml \
  --device cuda

# Evaluate with metrics + plots
!python src/eval/evaluate_model.py \
  --config configs/baseline_resnet50_regularized.yaml \
  --device cuda \
  --save-fig \
  --save-preds
```

### Download Results to Local Machine
```python
# In Colab
from google.colab import files

# Download checkpoint
files.download('checkpoints/baseline_resnet50_best.pth')

# Download metrics
files.download('runs/baseline_resnet50/metrics_test.json')
files.download('runs/baseline_resnet50/confusion_matrix_test.png')
files.download('runs/baseline_resnet50/roc_curves_test.png')

# Or download everything
!zip -r baseline_results.zip checkpoints/baseline_resnet50_* runs/baseline_resnet50/
files.download('baseline_results.zip')
```

---

**Report Generated:** November 18, 2025  
**Status:** Training interrupted at Epoch 3 (validation 10% complete)  
**Best Model:** Epoch 1, Val F1 = 0.5772  
**Ready for:** Test evaluation and professor presentation
