# NTHU Driver Drowsiness Detection with ROI

Image-based driver drowsiness detection on the NTHU-DDD2 dataset with subject-exclusive splits. Explores ROI-aware CNNs using pseudo eye/mouth segmentation masks, compact backbones (ResNet/EfficientNet/VGG), and robustness to glare, blur, and eyelid occlusion. Includes clean manifests, train scripts, and reproducible experiments.

## Features

- **Modular Architecture**: Clean `src/` layout with separate modules for models, data, and utilities
- **Multiple Backbones**: Support for ResNet, EfficientNet, and VGG architectures
- **ROI-Aware Models**: Specialized attention mechanisms for eye and mouth regions
- **Config-Driven**: YAML-based configuration system for reproducible experiments
- **Subject-Exclusive Splits**: Proper evaluation protocol avoiding data leakage
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, and AUC tracking
- **Visualization Tools**: Training curves, confusion matrices, and ROC curves

## Project Structure

```
nthu-driver-drowsiness-ROI/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── builder.py          # Model factory
│   │   ├── resnet.py           # ResNet models
│   │   ├── efficientnet.py     # EfficientNet models
│   │   └── vgg.py              # VGG models
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # NTHU-DDD2 dataset loader
│   │   ├── roi_masks.py        # ROI mask generation
│   │   └── transforms.py       # Data augmentation
│   └── utils/
│       ├── __init__.py
│       ├── config.py           # Configuration manager
│       ├── metrics.py          # Metrics calculation
│       └── visualization.py    # Plotting utilities
├── configs/                    # Configuration files
│   ├── resnet18_baseline.yaml
│   ├── resnet18_roi.yaml
│   └── efficientnet_b0.yaml
├── experiments/                # Experiment outputs
├── train.py                    # Training script
├── eval.py                     # Evaluation script
├── requirements.txt            # Dependencies
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/hmolhem/nthu-driver-drowsiness-ROI.git
cd nthu-driver-drowsiness-ROI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Setup

The project expects the NTHU-DDD2 dataset in the following structure:

```
data/NTHU-DDD2/
├── subject_01/
│   ├── alert/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── drowsy/
│       ├── img_001.jpg
│       └── ...
├── subject_02/
│   └── ...
└── ...
```

## Usage

### Training

Train a model using a configuration file:

```bash
python train.py --config configs/resnet18_baseline.yaml
```

Or override config parameters from command line:

```bash
python train.py \
    --config configs/resnet18_baseline.yaml \
    --data-root ./data/NTHU-DDD2 \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.0001 \
    --backbone resnet50
```

### Evaluation

Evaluate a trained model:

```bash
python eval.py \
    --checkpoint experiments/resnet18_baseline/best_model.pth \
    --split test \
    --output-dir experiments/resnet18_baseline/results
```

### Configuration

Example configuration file (`configs/resnet18_baseline.yaml`):

```yaml
experiment:
  name: resnet18_baseline
  seed: 42

data:
  dataset_root: ./data/NTHU-DDD2
  image_size: [224, 224]
  batch_size: 32
  num_workers: 4
  augmentation: true

model:
  backbone: resnet18
  num_classes: 2
  pretrained: true
  use_roi: false
  dropout: 0.5

training:
  epochs: 50
  learning_rate: 0.001
  optimizer: adam
  scheduler: step
  step_size: 10
  gamma: 0.1
  early_stopping_patience: 10

logging:
  use_tensorboard: true
  save_frequency: 5
  log_frequency: 10
```

## Supported Models

### Backbones

- **ResNet**: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
- **EfficientNet**: `efficientnet_b0` through `efficientnet_b7`
- **VGG**: `vgg11`, `vgg13`, `vgg16`, `vgg19`

### ROI-Aware Variants

Set `use_roi: true` in config to use ROI attention mechanisms that focus on eye and mouth regions.

## ROI Mask Generation

Generate ROI masks for your dataset:

```python
from src.data.roi_masks import batch_generate_masks

batch_generate_masks(
    image_dir='./data/NTHU-DDD2/subject_01/alert',
    output_dir='./data/roi_masks/subject_01/alert',
    mask_type='eye_mouth'
)
```

## Key Features

### Subject-Exclusive Splits

The dataset loader automatically creates subject-exclusive train/val/test splits (70%/15%/15% by default) to prevent data leakage and ensure robust evaluation.

### Data Augmentation

Training includes:
- Random horizontal flip
- Random rotation (±10°)
- Color jitter (brightness, contrast, saturation, hue)
- Random affine transformations

### Metrics Tracking

Comprehensive metrics tracked during training:
- Training/validation loss
- Training/validation accuracy
- Precision, Recall, F1 Score
- ROC-AUC
- Confusion matrix
- ROC curves

### TensorBoard Integration

Monitor training in real-time:

```bash
tensorboard --logdir experiments/resnet18_baseline/runs
```

## Experiment Outputs

Each experiment creates:
- `config.yaml`: Saved configuration
- `best_model.pth`: Best model checkpoint
- `checkpoint_epoch_*.pth`: Periodic checkpoints
- `training_curves.png`: Loss and accuracy plots
- `runs/`: TensorBoard logs

## Example Experiments

### Baseline ResNet18

```bash
python train.py --config configs/resnet18_baseline.yaml
```

### ResNet18 with ROI Attention

```bash
python train.py --config configs/resnet18_roi.yaml
```

### EfficientNet-B0

```bash
python train.py --config configs/efficientnet_b0.yaml
```

## Development

### Adding New Models

1. Create model class in `src/models/`
2. Add to `src/models/__init__.py`
3. Update `build_model()` in `src/models/builder.py`

### Custom Metrics

Add metrics to `src/utils/metrics.py` and update `calculate_metrics()`.

### Custom Visualizations

Add plotting functions to `src/utils/visualization.py`.

## Citation

If you use this code, please cite:

```bibtex
@software{nthu_drowsiness_roi,
  title={NTHU Driver Drowsiness Detection with ROI},
  author={Your Name},
  year={2024},
  url={https://github.com/hmolhem/nthu-driver-drowsiness-ROI}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- NTHU-DDD2 dataset creators
- PyTorch and torchvision teams
- Pre-trained model providers
