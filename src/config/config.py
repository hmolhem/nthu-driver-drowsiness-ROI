"""Configuration management for NTHU-DDD2 driver drowsiness detection"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import yaml
from pathlib import Path


@dataclass
class DataConfig:
    """Dataset configuration"""
    data_root: str = "data/NTHU-DDD2"
    image_size: tuple = (224, 224)
    batch_size: int = 32
    num_workers: int = 4
    train_subjects: Optional[List[int]] = None
    val_subjects: Optional[List[int]] = None
    test_subjects: Optional[List[int]] = None
    use_roi: bool = True
    augmentation: bool = True
    

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    backbone: str = "resnet18"  # resnet18, resnet50, efficientnet_b0, vgg16
    pretrained: bool = True
    num_classes: int = 2  # awake vs drowsy
    dropout: float = 0.5
    use_roi_attention: bool = True
    freeze_backbone: bool = False
    

@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adam"  # adam, sgd, adamw
    scheduler: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 5
    early_stopping_patience: int = 15
    gradient_clip: Optional[float] = 1.0
    mixed_precision: bool = True
    

@dataclass
class ExperimentConfig:
    """Experiment tracking configuration"""
    experiment_name: str = "baseline"
    output_dir: str = "experiments"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_freq: int = 5
    eval_freq: int = 1
    use_wandb: bool = False
    wandb_project: str = "nthu-drowsiness"
    seed: int = 42
    

@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            experiment=ExperimentConfig(**config_dict.get('experiment', {}))
        )
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'experiment': self.experiment.__dict__
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def __str__(self) -> str:
        """String representation of config"""
        lines = ["Configuration:"]
        for section_name in ['data', 'model', 'training', 'experiment']:
            section = getattr(self, section_name)
            lines.append(f"\n{section_name.upper()}:")
            for key, value in section.__dict__.items():
                lines.append(f"  {key}: {value}")
        return '\n'.join(lines)
