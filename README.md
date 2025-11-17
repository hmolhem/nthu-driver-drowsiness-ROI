# NTHU Driver Drowsiness Detection with ROI-based Approach

Deep learning system for detecting driver drowsiness using facial ROI (Region of Interest) analysis with subject-exclusive evaluation to prevent data leakage.

## Quick Start

### 1. Setup Environment

```bash
# Activate virtual environment (auto-activated in VS Code)
.venv\Scripts\activate

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### 2. Prepare Dataset Splits

```bash
# Create subject-exclusive train/val/test splits
python src/data/create_splits.py

# Output: data/splits/{train,val,test}.csv
```

### 3. Train Baseline Model

```bash
# Train ResNet50 baseline
python src/training/train_baseline.py --config configs/baseline_resnet50.yaml

# Or train EfficientNet baseline  
python src/training/train_baseline.py --config configs/baseline_efficientnet.yaml --device cpu
```

## Project Structure

```text
├── configs/              # Training configurations
├── data/
│   ├── manifests/       # Dataset manifests (66,521 images)
│   └── splits/          # Train/val/test splits
├── datasets/archive/    # Raw dataset (drowsy/notdrowsy)
├── docs/                # Documentation
├── src/
│   ├── data/           # Dataset and transforms
│   ├── models/         # Model architectures
│   ├── training/       # Training engine and metrics
│   └── utils/          # Utilities
└── checkpoints/        # Saved models (generated)
```

## Dataset

- **Total Images**: 66,521
- **Subjects**: 4 (subject-exclusive splits)
- **Classes**: drowsy (54.2%), notdrowsy (45.8%)
- **Behaviors**: sleepyCombination, nonsleepyCombination, slowBlinkWithNodding, yawning
- **Splits**: Train (2 subjects), Val (1 subject), Test (1 subject)

## Models

### Baseline Models (Implemented)

- **ResNet50**: Standard CNN baseline
- **EfficientNet-B0**: Lightweight alternative

### ROI-based Models (Planned)

- ROI gating with attention mechanisms
- Multi-region feature extraction (eyes, mouth)
- Multi-task learning (classification + segmentation)

## Key Features

- ✅ **Subject-Exclusive Splits**: Prevents data leakage
- ✅ **Balanced Training**: Class-weighted loss
- ✅ **Comprehensive Metrics**: Macro-F1, per-class metrics, confusion matrix
- ✅ **Data Augmentation**: Color jitter, rotation, affine transforms
- ✅ **Early Stopping**: Monitors validation macro-F1
- ✅ **Learning Rate Scheduling**: ReduceLROnPlateau

## Documentation

- [Folder Structure](docs/folder-structure.md) - Project organization
- [Dataset Manifest](docs/manifest.md) - Dataset details and usage
- [Environment Setup](docs/env-setup.md) - Installation guide
- [Project Proposal](docs/proposal.md) - Full research proposal

## Requirements

- Python 3.10.11
- PyTorch 2.1.2
- TensorFlow 2.15.0 (for future experiments)
- See [requirements.txt](requirements.txt) for full list

## License

Research project for NTHU Driver Drowsiness Detection.

## Citation

If you use this code, please cite:

```bibtex
@misc{nthu-drowsiness-roi,
  title={Driver Drowsiness Detection with ROI-based Analysis},
  author={Your Name},
  year={2025}
}
```
