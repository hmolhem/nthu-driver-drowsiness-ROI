# CONTRIBUTING.md

Thank you for considering contributing to the NTHU Driver Drowsiness Detection project!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/nthu-driver-drowsiness-ROI.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes
6. Commit: `git commit -m "Add your feature"`
7. Push: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements.txt

# Install dev dependencies
pip install pytest black flake8 mypy
```

## Code Style

We follow PEP 8 guidelines. Please format your code with:

```bash
# Format code
black src/ *.py

# Check linting
flake8 src/ *.py --max-line-length=100
```

## Project Structure Guidelines

- **src/config/**: Configuration and parameter management
- **src/dataset/**: Data loading, preprocessing, augmentation
- **src/models/**: Neural network architectures
- **src/roi/**: ROI mask generation and processing
- **src/utils/**: Metrics, visualization, helper functions
- **experiments/configs/**: Experiment YAML configurations
- Root scripts: High-level training/evaluation scripts

## Adding New Features

### Adding a New Model Backbone

1. The framework uses `timm` for backbones, so most models are already supported
2. Test with `create_model(backbone='your_model_name')`
3. Add example config in `experiments/configs/`
4. Update README with model description

### Adding New Metrics

1. Add metric computation in `src/utils/metrics.py`
2. Update `MetricsCalculator.compute()` method
3. Add visualization if needed in `src/utils/visualization.py`
4. Update documentation

### Adding New Augmentations

1. Modify `src/dataset/augmentation.py`
2. Use albumentations library format
3. Test on sample images
4. Update documentation

## Testing

Before submitting a PR:

```bash
# Syntax check
python -m py_compile src/**/*.py *.py

# Test imports
python -c "from src.config.config import Config; Config()"

# Run examples
python examples.py

# Test with actual data (if available)
python train.py --config experiments/configs/baseline.yaml
```

## Documentation

- Update README.md for user-facing changes
- Update TESTING.md for testing procedures
- Add docstrings to new functions/classes
- Include type hints where possible

## Pull Request Guidelines

1. **Clear Description**: Explain what and why
2. **Single Purpose**: One feature/fix per PR
3. **Tests**: Include tests if applicable
4. **Documentation**: Update docs for user-facing changes
5. **Clean History**: Squash commits if needed
6. **No Breaking Changes**: Or clearly document them

## Commit Messages

Follow conventional commits format:

```
feat: Add ResNet101 support
fix: Correct ROI mask generation for edge cases
docs: Update README with new examples
refactor: Simplify data loader code
test: Add unit tests for metrics module
```

## Code Review Process

1. Automated checks must pass
2. At least one maintainer approval required
3. Address review comments
4. Squash and merge

## Areas for Contribution

### High Priority
- [ ] Add more data augmentation strategies
- [ ] Implement ensemble methods
- [ ] Add TensorBoard integration
- [ ] Create Jupyter notebook tutorials
- [ ] Add more visualization options

### Medium Priority
- [ ] Multi-GPU training support
- [ ] ONNX export for deployment
- [ ] Add more backbone options
- [ ] Implement attention visualization
- [ ] Create Docker container

### Nice to Have
- [ ] Web demo
- [ ] Mobile deployment
- [ ] Real-time inference
- [ ] Additional datasets support
- [ ] Hyperparameter optimization

## Questions?

Feel free to open an issue for:
- Bug reports
- Feature requests
- Documentation improvements
- General questions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
