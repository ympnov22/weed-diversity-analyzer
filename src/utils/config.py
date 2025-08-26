"""Configuration management utilities."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class PathConfig:
    """Configuration for file paths."""
    data_root: str
    raw_data: str
    processed_data: str
    models: str
    output: str
    logs: str
    temp: str


@dataclass
class ModelConfig:
    """Configuration for model settings."""
    name: str
    model_path: str
    confidence_threshold: float
    top_k: Optional[int] = None


class ConfigManager:
    """Manages application configuration from YAML files."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            self.config_path = project_root / "config" / "config.yaml"
        else:
            self.config_path = Path(config_path)
        self._config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
    
    def _validate_config(self) -> None:
        """Validate required configuration sections."""
        required_sections = ['app', 'paths', 'preprocessing', 'models', 'diversity']
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required configuration section: {section}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key.
        
        Args:
            key: Dot-separated key (e.g., 'models.primary.name')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_paths(self) -> PathConfig:
        """Get path configuration."""
        paths = self._config['paths']
        return PathConfig(**paths)
    
    def get_primary_model(self) -> ModelConfig:
        """Get primary model configuration."""
        model_config = self._config['models']['primary'].copy()
        # Remove fields that aren't part of ModelConfig dataclass
        for field in ['size', 'use_lora', 'lora_path', 'device', 'batch_size']:
            model_config.pop(field, None)
        return ModelConfig(**model_config)
    
    def get_fallback_models(self) -> list[ModelConfig]:
        """Get fallback model configurations."""
        fallback_configs = self._config['models']['fallback']
        models = []
        for config in fallback_configs:
            config_copy = config.copy()
            # Remove fields that aren't part of ModelConfig dataclass
            for field in ['size', 'use_lora', 'lora_path', 'device', 'batch_size']:
                config_copy.pop(field, None)
            models.append(ModelConfig(**config_copy))
        return models
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self._config['preprocessing']
    
    def get_diversity_config(self) -> Dict[str, Any]:
        """Get diversity analysis configuration."""
        return self._config['diversity']
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self._config['output']
    
    def create_directories(self) -> None:
        """Create necessary directories based on configuration."""
        paths = self.get_paths()
        directories = [
            paths.data_root,
            paths.raw_data,
            paths.processed_data,
            paths.models,
            paths.output,
            paths.logs,
            paths.temp,
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def update_config(self, key: str, value: Any) -> None:
        """Update configuration value.
        
        Args:
            key: Dot-separated key
            value: New value
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self, path: Optional[str] = None) -> None:
        """Save configuration to file.
        
        Args:
            path: Path to save to. If None, uses original path.
        """
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get full configuration dictionary."""
        return self._config.copy()
