# Keras/TensorFlow to PyTorch Quick Reference

This guide helps you transition from Keras/TensorFlow to PyTorch for this project.

## Core Concepts Comparison

| Concept | Keras/TensorFlow | PyTorch |
|---------|------------------|---------|
| **Model Class** | `keras.Model` | `torch.nn.Module` |
| **Layer Definition** | `model.add(Dense(64))` | `self.fc = nn.Linear(64)` |
| **Forward Pass** | `model.call()` (automatic) | `model.forward()` (you write it) |
| **Training** | `model.fit()` (automatic) | Write your own loop |
| **Evaluation** | `model.evaluate()` | Write your own loop |
| **Prediction** | `model.predict()` | `model(x)` with `torch.no_grad()` |

## 1. Model Definition

### Keras Way
```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation='softmax')
])
```

### PyTorch Way
```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # No softmax! CrossEntropyLoss includes it
        return x
```

**Key Difference:** In PyTorch, you define layers in `__init__` and connect them in `forward()`.

## 2. Training Loop

### Keras Way
```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_data,
    epochs=50,
    validation_data=val_data,
    callbacks=[early_stopping]
)
```

### PyTorch Way
```python
model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    # Training
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()      # Clear gradients
        outputs = model(images)    # Forward pass
        loss = criterion(outputs, labels)
        loss.backward()            # Compute gradients
        optimizer.step()           # Update weights
    
    # Validation
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            # Calculate metrics...
```

**Key Difference:** PyTorch gives you explicit control over each step. Keras does it all automatically.

## 3. Data Loading

### Keras Way
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10
)

train_gen = datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
```

### PyTorch Way
```python
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Define transforms (like ImageDataGenerator)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),  # Scales to [0,1] like rescale=1./255
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Create dataset
class MyDataset(Dataset):
    def __getitem__(self, idx):
        image = load_image(idx)
        image = self.transform(image)
        return image, label

# Create dataloader (like flow_from_directory)
train_loader = DataLoader(
    MyDataset(transform=transform),
    batch_size=32,
    shuffle=True
)
```

**Key Difference:** PyTorch separates transforms from data loading. More modular!

## 4. Common Layers Translation

| Keras | PyTorch |
|-------|---------|
| `Dense(64)` | `nn.Linear(in_features, 64)` |
| `Conv2D(32, (3,3))` | `nn.Conv2d(in_channels, 32, kernel_size=3)` |
| `MaxPooling2D((2,2))` | `nn.MaxPool2d(2)` |
| `Dropout(0.5)` | `nn.Dropout(0.5)` |
| `BatchNormalization()` | `nn.BatchNorm2d(num_features)` |
| `Activation('relu')` | `nn.ReLU()` or `torch.relu()` |
| `Flatten()` | `torch.flatten(x, start_dim=1)` |
| `GlobalAveragePooling2D()` | `nn.AdaptiveAvgPool2d(1)` |

## 5. Loss Functions

| Keras | PyTorch |
|-------|---------|
| `'categorical_crossentropy'` | `nn.CrossEntropyLoss()` |
| `'binary_crossentropy'` | `nn.BCEWithLogitsLoss()` |
| `'mse'` | `nn.MSELoss()` |

**Important:** PyTorch's `CrossEntropyLoss` includes softmax! Don't apply softmax in your model.

## 6. Optimizers

| Keras | PyTorch |
|-------|---------|
| `Adam(lr=0.001)` | `optim.Adam(model.parameters(), lr=0.001)` |
| `SGD(lr=0.01, momentum=0.9)` | `optim.SGD(model.parameters(), lr=0.01, momentum=0.9)` |
| `RMSprop(lr=0.001)` | `optim.RMSprop(model.parameters(), lr=0.001)` |

## 7. Model Training Modes

### Keras Way
```python
# Training mode (automatic with fit())
model.fit(...)

# Evaluation mode (automatic with evaluate())
model.evaluate(...)
```

### PyTorch Way
```python
# Training mode (enables dropout, batchnorm updates)
model.train()

# Evaluation mode (disables dropout, freezes batchnorm)
model.eval()
```

**Key Difference:** You must explicitly set training/eval mode in PyTorch!

## 8. Saving & Loading Models

### Keras Way
```python
# Save
model.save('model.h5')

# Load
model = keras.models.load_model('model.h5')
```

### PyTorch Way
```python
# Save
torch.save(model.state_dict(), 'model.pth')

# Load
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
```

## 9. GPU Usage

### Keras Way
```python
# Automatic GPU detection
with tf.device('/GPU:0'):
    model.fit(...)
```

### PyTorch Way
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model to GPU
model = model.to(device)

# Move data to GPU in training loop
for images, labels in dataloader:
    images = images.to(device)
    labels = labels.to(device)
```

## 10. In This Project

### What You Need to Know

**Good News:** Most PyTorch complexity is already handled for you!

1. **You mainly edit YAML configs** (like Keras hyperparameters)
2. **Training script is ready** (`train_baseline.py` - just run it!)
3. **Data loading is done** (`dataset.py` - handles everything)
4. **Models are defined** (`classifier.py` - ResNet50, EfficientNet)

### To Run Training (Simple!)

```bash
# Just like running a Keras script
python src/training/train_baseline.py --config configs/baseline_resnet50.yaml
```

### To Modify Hyperparameters

Edit `configs/baseline_resnet50.yaml`:
```yaml
training:
  batch_size: 32          # Like fit(batch_size=32)
  learning_rate: 0.0001   # Like Adam(lr=0.0001)
  epochs: 50              # Like fit(epochs=50)
```

### To Change Model Architecture

Edit config file:
```yaml
model:
  architecture: efficientnet_b0  # or resnet50, resnet18, etc.
  pretrained: true              # Like using ImageNet weights
  dropout: 0.5                  # Like Dropout(0.5)
```

## Tips for Keras Users

1. **Debugging:** PyTorch is easier to debug - you can print/inspect tensors anytime in the training loop
2. **Flexibility:** PyTorch gives more control for custom architectures (important for our ROI gating!)
3. **Learning curve:** Day 1-2 is unfamiliar, but by day 3 you'll appreciate the explicitness
4. **Documentation:** PyTorch docs are excellent - search "pytorch equivalent of keras X"
5. **This project:** You won't need to write much PyTorch code - just understand what's happening!

## Common Gotchas

1. **Always call `.to(device)`** for both model and data
2. **Always call `optimizer.zero_grad()`** before backward pass (Keras does this automatically)
3. **Always call `model.eval()`** before validation (Keras does this automatically)
4. **Don't use softmax** with CrossEntropyLoss (it's included!)
5. **Shape conventions:** PyTorch uses (N, C, H, W) while TensorFlow uses (N, H, W, C)

## Need Help?

The code in this project has inline comments explaining PyTorch concepts in Keras terms. Check:
- `src/models/classifier.py` - Model definition
- `src/training/trainer.py` - Training loop
- `src/data/dataset.py` - Data loading
- `src/data/transforms.py` - Data augmentation

Happy PyTorching! 🔥
