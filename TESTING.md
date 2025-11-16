# Testing Guide

This document describes how to test the NTHU Driver Drowsiness Detection framework.

## Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Unit Tests

### Test Configuration Module

```python
from src.config.config import Config

# Test default config
config = Config()
print(config)

# Test YAML save/load
config.to_yaml('test_config.yaml')
loaded_config = Config.from_yaml('test_config.yaml')
assert config.model.backbone == loaded_config.model.backbone
```

### Test Dataset Module

```python
from src.dataset.loader import NTHUDDDDataset, create_subject_splits

# Test subject splits
train_subs, val_subs, test_subs = create_subject_splits(
    'data/NTHU-DDD2', seed=42
)
print(f"Train: {train_subs}")
print(f"Val: {val_subs}")
print(f"Test: {test_subs}")

# Test dataset (requires actual data)
# dataset = NTHUDDDDataset('data/NTHU-DDD2', split='train')
# print(f"Dataset size: {len(dataset)}")
```

### Test Model Module

```python
import torch
from src.models.drowsiness_detector import create_model

# Test model creation
model = create_model(backbone='resnet18', pretrained=False)
print(f"Model created successfully")

# Test forward pass
x = torch.randn(2, 3, 224, 224)
output = model(x)
assert output.shape == (2, 2)  # batch_size=2, num_classes=2
print(f"Forward pass successful: {output.shape}")
```

### Test Metrics Module

```python
import torch
from src.utils.metrics import MetricsCalculator

# Test metrics calculator
calc = MetricsCalculator()

predictions = torch.tensor([0, 1, 0, 1, 0])
labels = torch.tensor([0, 1, 0, 0, 1])
probabilities = torch.tensor([
    [0.9, 0.1],
    [0.2, 0.8],
    [0.7, 0.3],
    [0.6, 0.4],
    [0.3, 0.7]
])

calc.update(predictions, labels, probabilities)
metrics = calc.compute()
print(f"Metrics: {metrics}")
```

## Integration Tests

### Test Training Pipeline

```bash
# Test training with dummy data (requires actual dataset)
python train.py --config experiments/configs/baseline.yaml
```

### Test Evaluation Pipeline

```bash
# Test evaluation (requires trained model)
python eval.py \
    --config experiments/configs/baseline.yaml \
    --checkpoint experiments/baseline_resnet18/checkpoints/best_model.pth \
    --split test \
    --visualize
```

### Test ROI Mask Generation

```bash
# Test ROI generation (requires dataset)
python generate_roi_masks.py \
    --data-root data/NTHU-DDD2 \
    --output-dir data/NTHU-DDD2-ROI
```

### Test Manifest Generation

```bash
# Test manifest creation (requires dataset)
python generate_manifest.py \
    --data-root data/NTHU-DDD2 \
    --output manifest.json
```

## Manual Verification

### 1. Check Project Structure

```bash
tree -I '__pycache__|*.pyc|.git' -L 3
```

Expected structure:
- `src/` with all modules
- `experiments/configs/` with YAML files
- Root scripts: `train.py`, `eval.py`, etc.

### 2. Verify Python Syntax

```bash
python -m py_compile src/**/*.py *.py
```

### 3. Check Config Files

```bash
python -c "
import yaml
for f in ['experiments/configs/baseline.yaml', 
          'experiments/configs/efficientnet.yaml',
          'experiments/configs/resnet50.yaml']:
    with open(f) as file:
        yaml.safe_load(file)
    print(f'{f}: OK')
"
```

### 4. Test Import Statements

```bash
python -c "
from src.config.config import Config
from src.dataset.loader import NTHUDDDDataset
from src.dataset.augmentation import get_train_augmentation
from src.models.drowsiness_detector import create_model
from src.utils.metrics import MetricsCalculator
from src.utils.visualization import plot_training_curves
from src.utils.helpers import set_seed
print('All imports successful!')
"
```

## Example Workflow

### Complete training and evaluation workflow:

```bash
# 1. Prepare dataset
python generate_manifest.py --data-root data/NTHU-DDD2 --output data/manifest.json

# 2. Generate ROI masks (optional)
python generate_roi_masks.py \
    --data-root data/NTHU-DDD2 \
    --output-dir data/NTHU-DDD2-ROI

# 3. Train model
python train.py --config experiments/configs/baseline.yaml

# 4. Evaluate model
python eval.py \
    --config experiments/configs/baseline.yaml \
    --checkpoint experiments/baseline_resnet18/checkpoints/best_model.pth \
    --split test \
    --visualize

# 5. Check results
ls experiments/baseline_resnet18/logs/
ls experiments/baseline_resnet18/evaluation/
```

## Expected Outputs

After successful training, you should have:

1. **Checkpoints**: `experiments/{exp_name}/checkpoints/`
   - `best_model.pth`: Best validation model
   - `final_model.pth`: Final epoch model
   - `checkpoint_epoch_X.pth`: Periodic saves

2. **Logs**: `experiments/{exp_name}/logs/`
   - `training_history.json`: Metrics per epoch
   - `training_curves.png`: Loss and accuracy plots

3. **Evaluation**: `experiments/{exp_name}/evaluation/`
   - `test_metrics.json`: Test metrics
   - `test_confusion_matrix.png`: Confusion matrix
   - `test_roc_curve.png`: ROC curve
   - `test_predictions.png`: Sample predictions

## Troubleshooting

### ImportError: No module named 'torch'
```bash
pip install -r requirements.txt
```

### Dataset not found
- Ensure NTHU-DDD2 dataset is in `data/NTHU-DDD2/`
- Structure: `subject_XX/awake|drowsy/*.jpg`

### CUDA out of memory
- Reduce batch size in config
- Use smaller model backbone
- Disable mixed precision

### Poor performance
- Check class balance
- Verify data augmentation is enabled
- Try different learning rates
- Use pretrained weights

## Performance Benchmarks

Expected performance on NTHU-DDD2:

| Model | Accuracy | F1-Score | Parameters |
|-------|----------|----------|------------|
| ResNet18 | ~85-90% | ~0.85 | ~11M |
| ResNet50 | ~88-92% | ~0.88 | ~24M |
| EfficientNet-B0 | ~87-91% | ~0.87 | ~4M |

*Note: Actual results depend on dataset split and training configuration*
