# Google Colab Training Guide

This guide explains how to run training on Google Colab and bring results back to your local machine.

---

## What to Upload

### ✅ **Upload to Google Drive** (Recommended)

**Upload these folders:**
1. **`datasets/archive/`** (~5-10 GB)
   - `datasets/archive/drowsy/` (36,030 images)
   - `datasets/archive/notdrowsy/` (30,491 images)
   
**Do NOT need to upload:**
- ❌ `.venv/` (virtual environment - will create fresh in Colab)
- ❌ `.vscode/` (VS Code settings - not needed)
- ❌ `checkpoints/` (will be created during training)

### ✅ **Clone from GitHub** (Easiest!)

All code is already on GitHub, so just clone it:
```python
!git clone https://github.com/hmolhem/nthu-driver-drowsiness-ROI.git
```

---

## Option 1: GitHub Code + Drive Dataset (Recommended)

### Step 1: Upload Dataset to Google Drive

1. Create folder structure in Google Drive:
   ```
   My Drive/
   └── drowsiness-dataset/
       └── archive/
           ├── drowsy/      (upload 36,030 images)
           └── notdrowsy/   (upload 30,491 images)
   ```

2. Upload can take 30-60 minutes depending on internet speed

### Step 2: Use Colab Notebook (see below)

The notebook will:
- Clone code from GitHub
- Mount Google Drive for dataset
- Run training on GPU
- Save results back to Google Drive

---

## Option 2: Upload Everything to Google Drive

If you prefer to upload entire project:

1. Zip your local project (excluding `.venv/` and `.vscode/`)
2. Upload zip to Google Drive
3. Extract in Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   
   !unzip /content/drive/MyDrive/nthu-driver-drowsiness-ROI.zip -d /content/
   ```

---

## Colab Notebook Instructions

A complete Colab notebook is provided: `notebooks/colab_training.ipynb`

### What it does:

1. **Setup Environment**
   - Check GPU availability
   - Clone GitHub repo OR mount Drive
   - Install any missing packages

2. **Prepare Dataset**
   - Mount Google Drive
   - Link dataset to project folder
   - Verify data splits exist

3. **Run Training**
   - Train ResNet50 baseline
   - Train EfficientNet baseline (optional)
   - Display progress in real-time

4. **Save Results**
   - Checkpoint files (`.pth`)
   - Training logs
   - Metrics JSON
   - All saved to Google Drive

5. **Download Results**
   - Package results as zip
   - Download to local machine

---

## Files to Bring Back to Local Machine

After training completes, download these from Google Drive:

### **Essential Files:**

1. **Checkpoints** (model weights):
   - `checkpoints/baseline_resnet50_best.pth` (~90 MB)
   - `checkpoints/baseline_resnet50_last.pth` (~90 MB)
   - `checkpoints/baseline_efficientnet_best.pth` (~20 MB, if trained)

2. **Training Logs**:
   - `results/baseline_resnet50/train_log.txt`
   - `results/baseline_resnet50/metrics_train.json`
   - `results/baseline_resnet50/metrics_val.json`

3. **Metrics & Plots** (if generated):
   - `results/baseline_resnet50/confusion_matrix.png`
   - `results/baseline_resnet50/training_curves.png`

### **How to Download:**

**From Colab:**
```python
# Create zip of results
!zip -r results.zip checkpoints/ results/

# Download
from google.colab import files
files.download('results.zip')
```

**From Google Drive:**
- Navigate to `My Drive/drowsiness-results/`
- Right-click → Download

---

## Where to Place Downloaded Files Locally

Extract downloaded files to your local project:

```
nthu-driver-drowsiness-ROI/
├── checkpoints/
│   ├── baseline_resnet50_best.pth      ← Place here
│   ├── baseline_resnet50_last.pth      ← Place here
│   └── baseline_efficientnet_best.pth  ← Place here
└── results/
    └── baseline_resnet50/
        ├── train_log.txt                ← Place here
        ├── metrics_train.json           ← Place here
        ├── metrics_val.json             ← Place here
        ├── confusion_matrix.png         ← Place here
        └── training_curves.png          ← Place here
```

---

## Colab Resources & Limits

### **Free Tier:**
- **GPU:** Tesla T4 (16GB VRAM) - usually sufficient
- **RAM:** 12-13 GB
- **Disk:** 100+ GB
- **Runtime:** 12 hours max, then auto-disconnect
- **Daily limit:** ~12-15 hours of GPU usage

### **Expected Training Time:**
- ResNet50: ~30-45 minutes (50 epochs on GPU)
- EfficientNet-B0: ~20-30 minutes (50 epochs on GPU)
- Both models: ~1-1.5 hours total

### **Tips:**
- Keep Colab tab open (prevents early disconnect)
- Enable early stopping (already configured: patience=10)
- Check progress every 30 minutes
- If disconnected, training state is lost (must restart)

---

## Troubleshooting

### **GPU Not Available**
```python
# Check GPU
!nvidia-smi

# If no GPU, go to: Runtime → Change runtime type → GPU
```

### **Out of Memory**
Reduce batch size in config:
```yaml
training:
  batch_size: 16  # Changed from 32
```

### **Dataset Not Found**
Verify Drive mount path:
```python
import os
print(os.listdir('/content/drive/MyDrive/drowsiness-dataset/archive/'))
# Should show: ['drowsy', 'notdrowsy']
```

### **Training Too Slow**
- Confirm GPU is active (check `nvidia-smi`)
- Reduce image size: `image_size: 192` instead of 224
- Use smaller model: EfficientNet-B0 instead of ResNet50

---

## Next Steps After Training

Once you have results back on your local machine:

1. **Commit checkpoints** (use Git LFS for large files):
   ```bash
   git lfs track "*.pth"
   git add checkpoints/ results/
   git commit -m "Add trained baseline model checkpoints and results"
   ```

2. **Evaluate on test set** (I'll help you with this):
   ```python
   python src/eval/evaluate_model.py --checkpoint checkpoints/baseline_resnet50_best.pth
   ```

3. **Generate reports**:
   - Classification report per subject
   - Confusion matrix visualization
   - ROC curves
   - Failure case analysis

4. **Compare models**:
   - ResNet50 vs EfficientNet performance
   - Training time comparison
   - Model size vs accuracy trade-off

5. **Proceed to ROI models** (next phase)

---

## Summary Checklist

- [ ] Upload dataset to Google Drive (or prepare to clone from GitHub)
- [ ] Open Colab notebook: `notebooks/colab_training.ipynb`
- [ ] Run all cells in order
- [ ] Monitor training progress (~1-2 hours)
- [ ] Download results zip from Colab
- [ ] Extract to local `checkpoints/` and `results/` folders
- [ ] Verify checkpoint files exist locally
- [ ] Ready for next phase: evaluation and ROI models!

---

**Ready to train?** Open the Colab notebook and follow the steps! 🚀
