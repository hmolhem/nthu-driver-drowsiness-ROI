# NTHU Driver Drowsiness Detection (ROI-Aware)

Image-based driver drowsiness detection on the NTHU-DDD2 dataset with subject-exclusive splits. Explores ROI-aware CNNs using pseudo eye/mouth segmentation masks, compact backbones (ResNet/EfficientNet/VGG), and robustness to glare, blur, and eyelid occlusion. Includes clean manifests, train scripts, and reproducible experiments.

## Features

- **Modular Architecture**: Clean `src/` layout with separate modules for dataset, models, ROI processing, and utilities
- **Subject-Exclusive Splits**: Ensures robust evaluation by preventing subject leakage between train/val/test sets
- **ROI-Aware Models**: Attention mechanisms focusing on critical facial regions (eyes and mouth)
- **Multiple Backbones**: Support for ResNet, EfficientNet, VGG, and MobileNet architectures
- **Config-Driven Design**: YAML-based configuration for easy experiment management
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, AUC, and confusion matrices
- **Rich Visualizations**: Training curves, ROC curves, confusion matrices, and prediction samples
- **Data Augmentation**: Robust augmentations to handle glare, blur, and occlusion

## Project Structure

```
nthu-driver-drowsiness-ROI/
├── src/
│   ├── config/          # Configuration management
│   │   └── config.py
│   ├── dataset/         # Dataset loading and augmentation
│   │   ├── loader.py
│   │   └── augmentation.py
│   ├── models/          # Model architectures
│   │   └── drowsiness_detector.py
│   ├── roi/             # ROI mask generation
│   │   └── mask_generator.py
│   └── utils/           # Metrics, visualization, helpers
│       ├── metrics.py
│       ├── visualization.py
│       └── helpers.py
├── experiments/
│   └── configs/         # Experiment configurations
│       ├── baseline.yaml
│       ├── efficientnet.yaml
│       └── resnet50.yaml
├── train.py             # Training script
├── eval.py              # Evaluation script
├── generate_roi_masks.py # ROI mask generation
└── requirements.txt     # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/hmolhem/nthu-driver-drowsiness-ROI.git
cd nthu-driver-drowsiness-ROI
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset Setup

1. Download the NTHU-DDD2 dataset and organize it as follows:
```
data/NTHU-DDD2/
├── subject_01/
│   ├── awake/
│   │   ├── image_001.jpg
│   │   └── ...
│   └── drowsy/
│       ├── image_001.jpg
│       └── ...
├── subject_02/
│   └── ...
└── ...
```

2. (Optional) Generate ROI masks for enhanced performance:
```bash
python generate_roi_masks.py \
    --data-root data/NTHU-DDD2 \
    --output-dir data/NTHU-DDD2-ROI
```

## Usage

### Training

Train a model using a configuration file:

```bash
# Train with baseline ResNet18 configuration
python train.py --config experiments/configs/baseline.yaml

# Train with EfficientNet
python train.py --config experiments/configs/efficientnet.yaml

# Train with ResNet50
python train.py --config experiments/configs/resnet50.yaml
```

The training script will:
- Automatically create subject-exclusive splits
- Save checkpoints and logs to `experiments/{experiment_name}/`
- Generate training curves and metrics
- Apply early stopping based on validation accuracy

### Evaluation

Evaluate a trained model:

```bash
python eval.py \
    --config experiments/configs/baseline.yaml \
    --checkpoint experiments/baseline_resnet18/checkpoints/best_model.pth \
    --split test \
    --visualize
```

This will:
- Compute comprehensive metrics (accuracy, precision, recall, F1, AUC)
- Generate confusion matrix and ROC curve
- Save visualizations of sample predictions
- Output detailed classification report

### Custom Configuration

Create your own configuration file in `experiments/configs/`:

```yaml
data:
  data_root: "data/NTHU-DDD2"
  image_size: [224, 224]
  batch_size: 32
  use_roi: true
  augmentation: true

model:
  backbone: "resnet18"  # resnet18, resnet50, efficientnet_b0, vgg16, etc.
  pretrained: true
  dropout: 0.5
  use_roi_attention: true

training:
  num_epochs: 100
  learning_rate: 0.0001
  optimizer: "adam"
  scheduler: "cosine"
  early_stopping_patience: 15

experiment:
  experiment_name: "my_experiment"
  seed: 42
```

## Model Architectures

The framework supports various CNN backbones through `timm`:

- **ResNet**: `resnet18`, `resnet34`, `resnet50`, `resnet101`
- **EfficientNet**: `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`, `efficientnet_b3`
- **VGG**: `vgg16`, `vgg19`
- **MobileNet**: `mobilenetv3_small`, `mobilenetv3_large`

All models can be augmented with ROI attention mechanisms to focus on eye and mouth regions.

## Data Augmentation

The augmentation pipeline includes:
- Geometric transformations (flip, rotate, scale, shift)
- Brightness and contrast adjustments
- Simulated glare (sun flare)
- Motion, Gaussian, and median blur
- Gaussian noise
- Random erasing (occlusion simulation)

## Metrics

The framework computes:
- **Overall**: Accuracy, weighted precision/recall/F1
- **Per-class**: Precision, recall, F1 for awake and drowsy classes
- **ROC**: AUC and ROC curve
- **Confusion Matrix**: Normalized and raw counts
- **Balanced Accuracy**: Average of per-class accuracies

## Experiment Tracking

Results are saved in `experiments/{experiment_name}/`:
- `config.yaml`: Experiment configuration
- `checkpoints/`: Model checkpoints
  - `best_model.pth`: Best validation accuracy
  - `final_model.pth`: Final epoch
  - `checkpoint_epoch_X.pth`: Periodic saves
- `logs/`: Training logs and visualizations
  - `training_history.json`: Loss and metrics per epoch
  - `training_curves.png`: Training/validation curves
- `evaluation/`: Evaluation results
  - `test_metrics.json`: Test set metrics
  - `test_confusion_matrix.png`
  - `test_roc_curve.png`
  - `test_predictions.png`

## Citation

If you use this code, please cite the NTHU-DDD dataset:

```bibtex
@article{weng2016driver,
  title={Driver drowsiness detection via a hierarchical temporal deep belief network},
  author={Weng, Che-Hsuan and Lai, Ying-Hsiu and Lai, Shang-Hong},
  journal={Computer Vision--ACCV 2016 Workshops},
  year={2016}
}
```

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
