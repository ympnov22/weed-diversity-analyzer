"""Tests for configuration management."""

import pytest
import tempfile
import yaml
from pathlib import Path
from src.utils.config import ConfigManager, PathConfig, ModelConfig


class TestConfigManager:
    """Test cases for ConfigManager."""
    
    def test_load_valid_config(self):
        """Test loading a valid configuration file."""
        config_data = {
            'app': {'name': 'test', 'version': '1.0'},
            'paths': {
                'data_root': 'data',
                'raw_data': 'data/raw',
                'processed_data': 'data/processed',
                'models': 'data/models',
                'output': 'output',
                'logs': 'logs',
                'temp': 'temp'
            },
            'preprocessing': {'image_size': [224, 224]},
            'models': {
                'primary': {
                    'name': 'weednet',
                    'model_path': 'models/weednet.pth',
                    'confidence_threshold': 0.5,
                    'top_k': 3
                },
                'fallback': []
            },
            'diversity': {'indices': ['richness', 'shannon']}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config_manager = ConfigManager(config_path)
            assert config_manager.get('app.name') == 'test'
            assert config_manager.get('app.version') == '1.0'
        finally:
            Path(config_path).unlink()
    
    def test_missing_config_file(self):
        """Test handling of missing configuration file."""
        with pytest.raises(FileNotFoundError):
            ConfigManager('/nonexistent/config.yaml')
    
    def test_invalid_yaml(self):
        """Test handling of invalid YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('invalid: yaml: content: [')
            config_path = f.name
        
        try:
            with pytest.raises(ValueError):
                ConfigManager(config_path)
        finally:
            Path(config_path).unlink()
    
    def test_get_with_default(self):
        """Test getting configuration values with defaults."""
        config_data = {
            'app': {'name': 'test'},
            'paths': {
                'data_root': 'data',
                'raw_data': 'data/raw',
                'processed_data': 'data/processed',
                'models': 'data/models',
                'output': 'output',
                'logs': 'logs',
                'temp': 'temp'
            },
            'preprocessing': {},
            'models': {'primary': {'name': 'test', 'model_path': 'test', 'confidence_threshold': 0.5}, 'fallback': []},
            'diversity': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config_manager = ConfigManager(config_path)
            assert config_manager.get('app.name') == 'test'
            assert config_manager.get('app.nonexistent', 'default') == 'default'
            assert config_manager.get('nonexistent.key', 42) == 42
        finally:
            Path(config_path).unlink()
    
    def test_path_config(self):
        """Test PathConfig creation."""
        config_data = {
            'app': {'name': 'test'},
            'paths': {
                'data_root': 'data',
                'raw_data': 'data/raw',
                'processed_data': 'data/processed',
                'models': 'data/models',
                'output': 'output',
                'logs': 'logs',
                'temp': 'temp'
            },
            'preprocessing': {},
            'models': {'primary': {'name': 'test', 'model_path': 'test', 'confidence_threshold': 0.5}, 'fallback': []},
            'diversity': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config_manager = ConfigManager(config_path)
            paths = config_manager.get_paths()
            assert isinstance(paths, PathConfig)
            assert paths.data_root == 'data'
            assert paths.raw_data == 'data/raw'
        finally:
            Path(config_path).unlink()
    
    def test_model_config(self):
        """Test ModelConfig creation."""
        config_data = {
            'app': {'name': 'test'},
            'paths': {
                'data_root': 'data',
                'raw_data': 'data/raw',
                'processed_data': 'data/processed',
                'models': 'data/models',
                'output': 'output',
                'logs': 'logs',
                'temp': 'temp'
            },
            'preprocessing': {},
            'models': {
                'primary': {
                    'name': 'weednet',
                    'model_path': 'models/weednet.pth',
                    'confidence_threshold': 0.7,
                    'top_k': 5
                },
                'fallback': [
                    {
                        'name': 'inatag',
                        'model_path': 'models/inatag.pth',
                        'confidence_threshold': 0.5
                    }
                ]
            },
            'diversity': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            config_manager = ConfigManager(config_path)
            
            primary_model = config_manager.get_primary_model()
            assert isinstance(primary_model, ModelConfig)
            assert primary_model.name == 'weednet'
            assert primary_model.confidence_threshold == 0.7
            assert primary_model.top_k == 5
            
            fallback_models = config_manager.get_fallback_models()
            assert len(fallback_models) == 1
            assert fallback_models[0].name == 'inatag'
            assert fallback_models[0].confidence_threshold == 0.5
        finally:
            Path(config_path).unlink()
