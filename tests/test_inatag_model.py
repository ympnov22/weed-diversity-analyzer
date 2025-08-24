"""Tests for iNatAg model implementation."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch

from src.models.inatag_model import iNatAgModel
from src.models.base_model import ModelConfig


class TestiNatAgModel:
    """Test cases for iNatAg model."""
    
    @pytest.fixture
    def model_config(self):
        """Create test model configuration."""
        return ModelConfig(
            model_name="test_inatag",
            model_path="test/models/inatag",
            confidence_threshold=0.5,
            top_k=3,
            device="cpu",
            batch_size=16
        )
    
    @pytest.fixture
    def inatag_model(self, model_config):
        """Create iNatAg model instance."""
        return iNatAgModel(model_config, model_size='tiny')
    
    def test_model_initialization(self, inatag_model):
        """Test model initialization."""
        assert inatag_model.model_size == 'tiny'
        assert inatag_model.repo_id == "Project-AgML/iNatAg-models"
        assert inatag_model.num_classes == 2959
        assert not inatag_model.is_loaded
    
    def test_available_sizes(self):
        """Test available model sizes."""
        expected_sizes = {'tiny', 'small', 'base', 'large'}
        assert set(iNatAgModel.AVAILABLE_SIZES.keys()) == expected_sizes
    
    def test_get_model_filename(self, inatag_model):
        """Test model filename generation."""
        filename = inatag_model._get_model_filename()
        assert filename == "swin_tiny_without_lora.pth"
        
        inatag_model.config.use_lora = True
        filename_lora = inatag_model._get_model_filename()
        assert filename_lora == "swin_tiny_with_lora.pth"
    
    @patch('src.models.inatag_model.hf_hub_download')
    @patch('src.models.inatag_model.timm.create_model')
    @patch('torch.load')
    def test_load_model_success(self, mock_torch_load, mock_timm_create, mock_hf_download, inatag_model):
        """Test successful model loading."""
        mock_hf_download.return_value = "/fake/path/model.pth"
        mock_model = Mock()
        mock_timm_create.return_value = mock_model
        mock_torch_load.return_value = {'fake': 'state_dict'}
        
        with patch.object(inatag_model, '_load_inatag_species_list', return_value=True):
            result = inatag_model.load_model()
        
        assert result is True
        assert inatag_model.is_loaded is True
        mock_hf_download.assert_called_once()
        mock_timm_create.assert_called_once()
        mock_model.load_state_dict.assert_called_once()
    
    @patch('src.models.inatag_model.hf_hub_download')
    def test_load_model_failure(self, mock_hf_download, inatag_model):
        """Test model loading failure."""
        mock_hf_download.side_effect = Exception("Download failed")
        
        result = inatag_model.load_model()
        
        assert result is False
        assert inatag_model.is_loaded is False
    
    def test_load_species_list_placeholder(self, inatag_model):
        """Test species list loading with placeholder generation."""
        with patch('src.models.inatag_model.hf_hub_download', side_effect=Exception("Not found")):
            result = inatag_model._load_inatag_species_list()
        
        assert result is True
        assert len(inatag_model.species_list) == 2959
        assert inatag_model.species_list[0] == "species_0000"
        assert inatag_model.species_list[100] == "species_0100"
    
    def test_preprocess_image(self, inatag_model):
        """Test image preprocessing."""
        test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        
        processed = inatag_model.preprocess_image(test_image)
        
        assert processed.shape == (3, 224, 224)
        assert processed.dtype == np.float32
        assert processed.min() >= -3.0
        assert processed.max() <= 3.0
    
    def test_preprocess_image_grayscale(self, inatag_model):
        """Test preprocessing of grayscale image."""
        test_image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        
        processed = inatag_model.preprocess_image(test_image)
        
        assert processed.shape == (3, 224, 224)
        assert processed.dtype == np.float32
    
    @patch('torch.no_grad')
    def test_predict_not_loaded(self, mock_no_grad, inatag_model):
        """Test prediction when model is not loaded."""
        with pytest.raises(RuntimeError, match="Model not properly loaded"):
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            inatag_model.predict(test_image)
    
    def test_get_model_info(self, inatag_model):
        """Test model info retrieval."""
        info = inatag_model.get_model_info()
        
        assert info['model_name'] == 'test_inatag'
        assert info['model_size'] == 'tiny'
        assert info['repo_id'] == 'Project-AgML/iNatAg-models'
        assert info['architecture'] == 'Swin Transformer'
        assert info['num_classes'] == 2959
    
    def test_device_selection_auto(self, model_config):
        """Test automatic device selection."""
        model = iNatAgModel(model_config)
        device = model._get_device()
        
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda']
    
    def test_device_selection_explicit(self, model_config):
        """Test explicit device selection."""
        model_config.device = "cpu"
        model = iNatAgModel(model_config)
        device = model._get_device()
        
        assert device.type == 'cpu'


@pytest.fixture
def mock_loaded_model():
    """Create a mock loaded model for testing predictions."""
    config = ModelConfig(
        model_name="test_inatag",
        model_path="test/models/inatag",
        confidence_threshold=0.5,
        top_k=3,
        device="cpu"
    )
    
    model = iNatAgModel(config, model_size='tiny')
    model.is_loaded = True
    model.species_list = [f"species_{i:04d}" for i in range(2959)]
    
    mock_torch_model = Mock()
    mock_torch_model.return_value = torch.randn(1, 2959)
    model.model = mock_torch_model
    model.device = torch.device('cpu')
    
    return model


class TestiNatAgModelPrediction:
    """Test prediction functionality."""
    
    def test_predict_success(self, mock_loaded_model):
        """Test successful prediction."""
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        with patch('torch.no_grad'):
            result = mock_loaded_model.predict(test_image)
        
        assert len(result.predictions) <= 3
        assert result.processing_time > 0
        assert result.model_info is not None
        assert len(result.confidence_scores) == len(result.predictions)
    
    def test_predict_batch_success(self, mock_loaded_model):
        """Test successful batch prediction."""
        test_images = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        
        mock_loaded_model.model.return_value = torch.randn(5, 2959)
        
        with patch('torch.no_grad'):
            results = mock_loaded_model.predict_batch(test_images)
        
        assert len(results) == 5
        for result in results:
            assert len(result.predictions) <= 3
            assert result.processing_time >= 0
