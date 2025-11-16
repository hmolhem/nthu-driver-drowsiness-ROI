"""Configuration utilities for the driver drowsiness detection project."""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for experiments."""
    
    def __init__(self, config_path: str = None, config_dict: Dict[str, Any] = None):
        """
        Initialize configuration from file or dictionary.
        
        Args:
            config_path: Path to YAML configuration file
            config_dict: Configuration dictionary (alternative to file)
        """
        if config_path is not None:
            self.config = self._load_yaml(config_path)
        elif config_dict is not None:
            self.config = config_dict
        else:
            self.config = self._default_config()
    
    @staticmethod
    def _load_yaml(path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'experiment': {
                'name': 'default_experiment',
                'seed': 42,
            },
            'data': {
                'dataset_root': './data/NTHU-DDD2',
                'image_size': [224, 224],
                'batch_size': 32,
                'num_workers': 4,
                'augmentation': True,
            },
            'model': {
                'backbone': 'resnet18',
                'num_classes': 2,
                'pretrained': True,
                'use_roi': True,
            },
            'training': {
                'epochs': 50,
                'learning_rate': 0.001,
                'optimizer': 'adam',
                'scheduler': 'step',
                'step_size': 10,
                'gamma': 0.1,
                'early_stopping_patience': 10,
            },
            'logging': {
                'use_tensorboard': True,
                'save_frequency': 5,
                'log_frequency': 10,
            }
        }
    
    def get(self, key: str, default=None):
        """Get configuration value by dot-separated key."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by dot-separated key."""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def save(self, path: str):
        """Save configuration to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def __repr__(self):
        return f"Config({yaml.dump(self.config, default_flow_style=False)})"
