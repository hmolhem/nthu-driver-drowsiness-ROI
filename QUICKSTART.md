# Quick Start Guide

Get up and running with NTHU Driver Drowsiness Detection in 5 minutes.

## Installation

```bash
# Clone repository
git clone https://github.com/hmolhem/nthu-driver-drowsiness-ROI.git
cd nthu-driver-drowsiness-ROI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Test (No Dataset Required)

Test the framework without a dataset:

```bash
# Run examples
python examples.py
```

This will:
- Create and test model architectures
- Test configuration management
- Compare different backbones
- Demonstrate data augmentation

## With Dataset

### 1. Prepare Dataset

Download NTHU-DDD2 dataset and organize as:

```
data/NTHU-DDD2/
├── subject_01/
│   ├── awake/
│   │   └── *.jpg
│   └── drowsy/
│       └── *.jpg
├── subject_02/
│   └── ...
```

### 2. Generate Manifest (Optional)

```bash
python generate_manifest.py \
    --data-root data/NTHU-DDD2 \
    --output data/manifest.json
```

### 3. Train Model

```bash
# Train with baseline ResNet18
python train.py --config experiments/configs/baseline.yaml
```

Training will:
- Create subject-exclusive splits automatically
- Save checkpoints to `experiments/baseline_resnet18/checkpoints/`
- Generate training curves in `experiments/baseline_resnet18/logs/`
- Apply early stopping

### 4. Evaluate Model

```bash
python eval.py \
    --config experiments/configs/baseline.yaml \
    --checkpoint experiments/baseline_resnet18/checkpoints/best_model.pth \
    --split test \
    --visualize
```

Results saved to `experiments/baseline_resnet18/evaluation/`

## Configuration

### Quick Config Modification

Edit `experiments/configs/baseline.yaml`:

```yaml
# Change batch size
data:
  batch_size: 16  # Default: 32

# Change model
model:
  backbone: "efficientnet_b0"  # Default: "resnet18"

# Adjust training
training:
  num_epochs: 50  # Default: 100
  learning_rate: 0.001  # Default: 0.0001
```

### Available Backbones

- ResNet: `resnet18`, `resnet34`, `resnet50`, `resnet101`
- EfficientNet: `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`
- VGG: `vgg16`, `vgg19`
- MobileNet: `mobilenetv3_small`, `mobilenetv3_large`

## Common Commands

```bash
# Train with custom config
python train.py --config path/to/config.yaml

# Evaluate on validation set
python eval.py --config CONFIG --checkpoint MODEL.pth --split val

# Generate ROI masks
python generate_roi_masks.py --data-root DATA --output-dir OUTPUT

# Create dataset manifest
python generate_manifest.py --data-root DATA --output manifest.json
```

## Expected Results

After training (100 epochs on NTHU-DDD2):

| Model | Accuracy | Training Time* |
|-------|----------|----------------|
| ResNet18 | ~85-90% | ~2-3 hours |
| ResNet50 | ~88-92% | ~4-5 hours |
| EfficientNet-B0 | ~87-91% | ~3-4 hours |

*On NVIDIA RTX 3090 with batch_size=32

## Troubleshooting

### Out of Memory
```yaml
# Reduce batch size in config
data:
  batch_size: 16  # or 8
```

### Slow Training
```yaml
# Enable mixed precision
training:
  mixed_precision: true

# Use more workers
data:
  num_workers: 8
```

### Poor Performance
```yaml
# Ensure augmentation is enabled
data:
  augmentation: true

# Use pretrained weights
model:
  pretrained: true
```

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Check [TESTING.md](TESTING.md) for testing procedures
- See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- Explore `examples.py` for code examples

## Support

- Open an issue for bugs or questions
- Check existing issues first
- Provide minimal reproducible example

## Resources

- **NTHU-DDD2 Dataset**: Driver drowsiness detection dataset
- **timm Library**: PyTorch image models
- **Albumentations**: Image augmentation library
