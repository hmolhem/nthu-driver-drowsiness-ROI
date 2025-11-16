# Quick Start Guide

This guide will help you get started with the NTHU Driver Drowsiness Detection project quickly.

## Setup

1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

2. **Verify Installation**

```bash
python test_setup.py
```

You should see:
```
============================================================
✓ All tests passed! Setup is working correctly.
============================================================
```

## Dataset Preparation

Place your NTHU-DDD2 dataset in the following structure:

```
data/NTHU-DDD2/
├── subject_01/
│   ├── alert/
│   │   └── *.jpg
│   └── drowsy/
│       └── *.jpg
├── subject_02/
│   └── ...
```

The dataset loader will automatically create subject-exclusive splits (70% train, 15% val, 15% test).

## Training Your First Model

### Option 1: Use a Pre-configured Model

```bash
python train.py --config configs/resnet18_baseline.yaml
```

### Option 2: Override Configuration Parameters

```bash
python train.py \
    --config configs/resnet18_baseline.yaml \
    --data-root ./data/NTHU-DDD2 \
    --epochs 30 \
    --batch-size 64 \
    --backbone resnet50
```

### Option 3: Train Without Config File

```bash
python train.py \
    --data-root ./data/NTHU-DDD2 \
    --backbone resnet18 \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001
```

## Monitoring Training

Training progress is automatically logged to TensorBoard. Monitor in real-time:

```bash
tensorboard --logdir experiments/resnet18_baseline/runs
```

Then open http://localhost:6006 in your browser.

## Evaluating a Model

After training, evaluate on the test set:

```bash
python eval.py \
    --checkpoint experiments/resnet18_baseline/best_model.pth \
    --split test \
    --output-dir experiments/resnet18_baseline/results
```

This will generate:
- `confusion_matrix_test.png`: Visual confusion matrix
- `roc_curve_test.png`: ROC curve with AUC score
- `metrics_test.txt`: Text file with all metrics

## Working with ROI Masks

### Generate ROI Masks (Optional)

If you want to use pre-generated ROI masks:

```bash
python generate_roi_masks.py \
    --data-root ./data/NTHU-DDD2 \
    --output-dir ./data/roi_masks \
    --mask-type eye_mouth
```

### Train with ROI Attention

Use the ROI-aware configuration:

```bash
python train.py --config configs/resnet18_roi.yaml
```

Or enable ROI in your own config by setting `use_roi: true`.

## Model Comparison

Train multiple models and compare:

```bash
# Baseline ResNet18
python train.py --config configs/resnet18_baseline.yaml

# ResNet18 with ROI
python train.py --config configs/resnet18_roi.yaml

# EfficientNet-B0
python train.py --config configs/efficientnet_b0.yaml

# VGG16
python train.py --config configs/vgg16_baseline.yaml
```

Evaluate each:

```bash
for exp in resnet18_baseline resnet18_roi efficientnet_b0 vgg16_baseline; do
    python eval.py \
        --checkpoint experiments/${exp}/best_model.pth \
        --split test \
        --output-dir experiments/${exp}/results
done
```

## Understanding Outputs

Each experiment creates:

```
experiments/your_experiment_name/
├── config.yaml              # Saved configuration
├── best_model.pth           # Best model checkpoint
├── checkpoint_epoch_*.pth   # Periodic checkpoints
├── training_curves.png      # Loss/accuracy plots
├── runs/                    # TensorBoard logs
└── results/                 # Evaluation outputs (if run)
    ├── confusion_matrix_test.png
    ├── roc_curve_test.png
    └── metrics_test.txt
```

## Configuration Parameters

Key parameters you can customize:

```yaml
experiment:
  name: my_experiment        # Experiment name
  seed: 42                   # Random seed

data:
  dataset_root: ./data/NTHU-DDD2
  image_size: [224, 224]     # Input image size
  batch_size: 32             # Batch size
  num_workers: 4             # DataLoader workers
  augmentation: true         # Enable augmentation

model:
  backbone: resnet18         # Model architecture
  num_classes: 2             # Alert/Drowsy
  pretrained: true           # Use ImageNet weights
  use_roi: false             # Enable ROI attention
  dropout: 0.5               # Dropout rate

training:
  epochs: 50                 # Training epochs
  learning_rate: 0.001       # Initial learning rate
  optimizer: adam            # adam or sgd
  scheduler: step            # step or cosine
  step_size: 10              # For step scheduler
  gamma: 0.1                 # LR decay factor
  early_stopping_patience: 10
```

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
python train.py --config configs/resnet18_baseline.yaml --batch-size 16
```

### Training Too Slow

- Reduce `num_workers` if CPU-bound
- Use smaller model (resnet18 vs resnet50)
- Use EfficientNet for efficiency

### Dataset Not Found

Make sure your dataset path matches what's in the config:
```bash
python train.py --data-root /path/to/your/NTHU-DDD2
```

## Next Steps

1. **Hyperparameter Tuning**: Experiment with learning rates, batch sizes, and schedulers
2. **Data Augmentation**: Adjust augmentation parameters in config
3. **Ensemble Methods**: Train multiple models and ensemble predictions
4. **Custom Architectures**: Add your own models to `src/models/`

## Support

For issues or questions:
1. Check the main README.md for detailed documentation
2. Review configuration files in `configs/`
3. Run `test_setup.py` to verify installation
4. Check training logs in `experiments/*/runs/`
