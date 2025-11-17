"""Configuration utilities for loading and validating config files."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Dictionary containing configuration
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save YAML file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


class DotDict(dict):
    """
    Dictionary subclass that allows attribute-style access.
    
    Example:
        config = DotDict({'model': {'name': 'resnet50'}})
        print(config.model.name)  # 'resnet50'
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")


def get_config(config_path: str) -> DotDict:
    """
    Load config and return as DotDict for easier access.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        DotDict object with configuration
    """
    config = load_config(config_path)
    return DotDict(config)
