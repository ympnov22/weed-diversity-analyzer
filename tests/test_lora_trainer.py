"""Tests for LoRA trainer implementation."""

import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.models.lora_trainer import LoRATrainer
from src.models.inatag_model import iNatAgModel
from src.models.base_model import ModelConfig


class TestLoRATrainer:
    """Test cases for LoRA trainer."""
    
    @pytest.fixture
    def mock_base_model(self):
        """Create mock base model."""
        config = ModelConfig(
            model_name="test_inatag",
            model_path="test/models/inatag",
            device="cpu"
        )
        
        model = iNatAgModel(config)
        model.is_loaded = True
        model.model = Mock()
        model.device = torch.device('cpu')
        
        return model
    
    @pytest.fixture
    def lora_trainer(self, mock_base_model):
        """Create LoRA trainer instance."""
        return LoRATrainer(mock_base_model)
    
    def test_initialization_default_config(self, lora_trainer):
        """Test LoRA trainer initialization with default config."""
        assert lora_trainer.lora_config['r'] == 16
        assert lora_trainer.lora_config['lora_alpha'] == 32
        assert lora_trainer.lora_config['lora_dropout'] == 0.1
        assert 'qkv' in lora_trainer.lora_config['target_modules']
        assert lora_trainer.peft_model is None
    
    def test_initialization_custom_config(self, mock_base_model):
        """Test LoRA trainer initialization with custom config."""
        custom_config = {
            'r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.05,
            'target_modules': ['attention']
        }
        
        trainer = LoRATrainer(mock_base_model, custom_config)
        
        assert trainer.lora_config['r'] == 8
        assert trainer.lora_config['lora_alpha'] == 16
        assert trainer.lora_config['lora_dropout'] == 0.05
        assert trainer.lora_config['target_modules'] == ['attention']
    
    @patch('src.models.lora_trainer.get_peft_model')
    @patch('src.models.lora_trainer.LoraConfig')
    def test_setup_lora_model_success(self, mock_lora_config, mock_get_peft_model, lora_trainer):
        """Test successful LoRA model setup."""
        mock_peft_model = Mock()
        mock_get_peft_model.return_value = mock_peft_model
        
        result = lora_trainer.setup_lora_model()
        
        assert result is True
        assert lora_trainer.peft_model == mock_peft_model
        mock_lora_config.assert_called_once()
        mock_get_peft_model.assert_called_once()
        mock_peft_model.print_trainable_parameters.assert_called_once()
    
    def test_setup_lora_model_not_loaded(self, lora_trainer):
        """Test LoRA setup when base model is not loaded."""
        lora_trainer.base_model.is_loaded = False
        
        result = lora_trainer.setup_lora_model()
        
        assert result is False
    
    @patch('cv2.imread')
    def test_prepare_training_data_success(self, mock_imread, lora_trainer):
        """Test successful training data preparation."""
        mock_imread.return_value = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        image_paths = [Path("test1.jpg"), Path("test2.jpg")]
        labels = [0, 1]
        
        with patch.object(lora_trainer.base_model, 'preprocess_image', return_value=np.random.randn(3, 224, 224)):
            images_tensor, labels_tensor = lora_trainer.prepare_training_data(image_paths, labels)
        
        assert images_tensor.shape[0] == 2
        assert labels_tensor.shape[0] == 2
        assert torch.is_tensor(images_tensor)
        assert torch.is_tensor(labels_tensor)
    
    @patch('cv2.imread')
    def test_prepare_training_data_no_valid_images(self, mock_imread, lora_trainer):
        """Test training data preparation with no valid images."""
        mock_imread.return_value = None
        
        image_paths = [Path("test1.jpg")]
        labels = [0]
        
        with pytest.raises(ValueError, match="No valid images found for training"):
            lora_trainer.prepare_training_data(image_paths, labels)
    
    def test_train_lora_no_peft_model(self, lora_trainer):
        """Test LoRA training without peft model setup."""
        train_images = torch.randn(4, 3, 224, 224)
        train_labels = torch.tensor([0, 1, 0, 1])
        
        with patch.object(lora_trainer, 'setup_lora_model', return_value=False):
            with pytest.raises(RuntimeError, match="Failed to setup LoRA model"):
                lora_trainer.train_lora(train_images, train_labels)
    
    @patch('torch.utils.data.DataLoader')
    @patch('torch.optim.AdamW')
    @patch('torch.nn.CrossEntropyLoss')
    def test_train_lora_success(self, mock_criterion, mock_optimizer, mock_dataloader, lora_trainer):
        """Test successful LoRA training."""
        lora_trainer.peft_model = Mock()
        lora_trainer.peft_model.parameters.return_value = []
        
        train_images = torch.randn(4, 3, 224, 224)
        train_labels = torch.tensor([0, 1, 0, 1])
        
        mock_train_loader = Mock()
        mock_train_loader.__iter__ = Mock(return_value=iter([(train_images[:2], train_labels[:2])]))
        mock_dataloader.return_value = mock_train_loader
        
        with patch.object(lora_trainer, '_train_epoch', return_value=(0.5, 0.8)):
            history = lora_trainer.train_lora(train_images, train_labels, epochs=1)
        
        assert 'train_loss' in history
        assert 'train_acc' in history
        assert len(history['train_loss']) == 1
        assert len(history['train_acc']) == 1
    
    def test_save_lora_adapter_no_model(self, lora_trainer):
        """Test saving LoRA adapter without model."""
        save_path = Path("test_adapter")
        
        result = lora_trainer.save_lora_adapter(save_path)
        
        assert result is False
    
    def test_save_lora_adapter_success(self, lora_trainer):
        """Test successful LoRA adapter saving."""
        lora_trainer.peft_model = Mock()
        save_path = Path("test_adapter")
        
        result = lora_trainer.save_lora_adapter(save_path)
        
        assert result is True
        lora_trainer.peft_model.save_pretrained.assert_called_once_with(str(save_path))
    
    @patch('src.models.lora_trainer.PeftModel.from_pretrained')
    def test_load_lora_adapter_success(self, mock_from_pretrained, lora_trainer):
        """Test successful LoRA adapter loading."""
        mock_loaded_model = Mock()
        mock_from_pretrained.return_value = mock_loaded_model
        
        load_path = Path("test_adapter")
        
        with patch.object(lora_trainer, 'setup_lora_model', return_value=True):
            result = lora_trainer.load_lora_adapter(load_path)
        
        assert result is True
        assert lora_trainer.peft_model == mock_loaded_model
        mock_from_pretrained.assert_called_once_with(
            lora_trainer.base_model.model, str(load_path)
        )
    
    def test_get_training_summary_no_history(self, lora_trainer):
        """Test training summary without training history."""
        summary = lora_trainer.get_training_summary()
        
        assert summary == {}
    
    def test_get_training_summary_with_history(self, lora_trainer):
        """Test training summary with training history."""
        lora_trainer.training_history = {
            'train_loss': [0.8, 0.6, 0.4],
            'train_acc': [0.6, 0.7, 0.8],
            'val_loss': [0.7, 0.5, 0.3],
            'val_acc': [0.65, 0.75, 0.85]
        }
        
        summary = lora_trainer.get_training_summary()
        
        assert summary['epochs'] == 3
        assert summary['final_train_loss'] == 0.4
        assert summary['final_train_acc'] == 0.8
        assert summary['final_val_loss'] == 0.3
        assert summary['final_val_acc'] == 0.85
        assert summary['best_val_acc'] == 0.85
        assert 'lora_config' in summary
