"""Tests for model manager implementation."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.models.model_manager import ModelManager
from src.models.base_model import PredictionResult
from src.utils.config import ConfigManager
from src.utils.data_structures import SpeciesPrediction


class TestModelManager:
    """Test cases for model manager."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration manager."""
        config = Mock()
        config.get_model_config = Mock(side_effect=lambda key, default=None: {
            'primary': {
                'name': 'inatag',
                'size': 'base',
                'model_path': 'data/models/inatag',
                'confidence_threshold': 0.7,
                'top_k': 3,
                'device': 'cpu'
            },
            'fallback': [
                {
                    'name': 'inatag_tiny',
                    'size': 'tiny',
                    'model_path': 'data/models/inatag',
                    'confidence_threshold': 0.5,
                    'top_k': 3,
                    'device': 'cpu'
                }
            ]
        }.get(key, default))
        
        return config
    
    @pytest.fixture
    def model_manager(self, mock_config):
        """Create model manager instance."""
        return ModelManager(mock_config)
    
    def test_initialization(self, model_manager):
        """Test model manager initialization."""
        assert model_manager.primary_model is None
        assert model_manager.fallback_models == []
        assert model_manager.model_performance == {}
    
    @patch('src.models.model_manager.iNatAgModel')
    def test_load_models_success(self, mock_inatag_model, model_manager):
        """Test successful model loading."""
        mock_primary = Mock()
        mock_primary.load_model.return_value = True
        mock_fallback = Mock()
        mock_fallback.load_model.return_value = True
        
        mock_inatag_model.side_effect = [mock_primary, mock_fallback]
        
        result = model_manager.load_models()
        
        assert result is True
        assert model_manager.primary_model == mock_primary
        assert len(model_manager.fallback_models) == 1
        assert model_manager.fallback_models[0] == mock_fallback
    
    @patch('src.models.model_manager.iNatAgModel')
    def test_load_models_primary_failure(self, mock_inatag_model, model_manager):
        """Test model loading with primary model failure."""
        mock_primary = Mock()
        mock_primary.load_model.return_value = False
        mock_inatag_model.return_value = mock_primary
        
        result = model_manager.load_models()
        
        assert result is False
    
    def test_predict_with_fallback_reliable_primary(self, model_manager):
        """Test prediction with reliable primary model."""
        mock_primary = Mock()
        mock_prediction = Mock(spec=PredictionResult)
        mock_prediction.processing_time = 0.1
        mock_prediction.predictions = [Mock(confidence=0.8)]
        mock_primary.predict.return_value = mock_prediction
        mock_primary.is_prediction_reliable.return_value = True
        
        model_manager.primary_model = mock_primary
        
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = model_manager.predict_with_fallback(test_image)
        
        assert result == mock_prediction
        mock_primary.predict.assert_called_once_with(test_image)
    
    def test_predict_with_fallback_unreliable_primary(self, model_manager):
        """Test prediction with unreliable primary model using fallback."""
        mock_primary = Mock()
        mock_primary_prediction = Mock(spec=PredictionResult)
        mock_primary_prediction.processing_time = 0.1
        mock_primary_prediction.predictions = [Mock(confidence=0.6)]
        mock_primary.predict.return_value = mock_primary_prediction
        mock_primary.is_prediction_reliable.return_value = False
        
        mock_fallback = Mock()
        mock_fallback_prediction = Mock(spec=PredictionResult)
        mock_fallback_prediction.processing_time = 0.2
        mock_fallback_prediction.predictions = [Mock(confidence=0.9)]
        mock_fallback.predict.return_value = mock_fallback_prediction
        mock_fallback.is_prediction_reliable.return_value = True
        
        model_manager.primary_model = mock_primary
        model_manager.fallback_models = [mock_fallback]
        
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = model_manager.predict_with_fallback(test_image)
        
        assert result == mock_fallback_prediction
        mock_primary.predict.assert_called_once_with(test_image)
        mock_fallback.predict.assert_called_once_with(test_image)
    
    def test_predict_with_fallback_no_models(self, model_manager):
        """Test prediction without loaded models."""
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        with pytest.raises(RuntimeError, match="No models loaded"):
            model_manager.predict_with_fallback(test_image)
    
    def test_predict_batch_with_fallback(self, model_manager):
        """Test batch prediction with fallback."""
        mock_primary = Mock()
        
        reliable_result = Mock(spec=PredictionResult)
        unreliable_result = Mock(spec=PredictionResult)
        
        mock_primary.predict_batch.return_value = [reliable_result, unreliable_result]
        mock_primary.is_prediction_reliable.side_effect = [True, False]
        
        mock_fallback = Mock()
        fallback_result = Mock(spec=PredictionResult)
        mock_fallback.predict_batch.return_value = [fallback_result]
        mock_fallback.is_prediction_reliable.return_value = True
        
        model_manager.primary_model = mock_primary
        model_manager.fallback_models = [mock_fallback]
        
        test_images = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        ]
        
        results = model_manager.predict_batch_with_fallback(test_images)
        
        assert len(results) == 2
        assert results[0] == reliable_result
        assert results[1] == fallback_result
    
    def test_soft_voting_aggregation(self, model_manager):
        """Test soft voting aggregation."""
        prediction1 = SpeciesPrediction(
            species_name="species_a", confidence=0.8,
            scientific_name="Species A", common_name=""
        )
        prediction2 = SpeciesPrediction(
            species_name="species_b", confidence=0.6,
            scientific_name="Species B", common_name=""
        )
        
        result1 = PredictionResult(
            predictions=[prediction1],
            processing_time=0.1,
            model_info={'top_k': 3},
            confidence_scores=[0.8],
            raw_outputs=np.array([0.8, 0.2])
        )
        
        result2 = PredictionResult(
            predictions=[prediction2],
            processing_time=0.1,
            model_info={'top_k': 3},
            confidence_scores=[0.6],
            raw_outputs=np.array([0.4, 0.6])
        )
        
        aggregated = model_manager.aggregate_predictions([result1, result2], method='soft_voting')
        
        assert aggregated.model_info['aggregation_method'] == 'soft_voting'
        assert aggregated.model_info['num_models'] == 2
        assert aggregated.raw_outputs is not None
    
    def test_hard_voting_aggregation(self, model_manager):
        """Test hard voting aggregation."""
        prediction1 = SpeciesPrediction(
            species_name="species_a", confidence=0.8,
            scientific_name="Species A", common_name=""
        )
        prediction2 = SpeciesPrediction(
            species_name="species_a", confidence=0.7,
            scientific_name="Species A", common_name=""
        )
        
        result1 = PredictionResult(
            predictions=[prediction1],
            processing_time=0.1,
            model_info={'top_k': 3},
            confidence_scores=[0.8],
            raw_outputs=None
        )
        
        result2 = PredictionResult(
            predictions=[prediction2],
            processing_time=0.1,
            model_info={'top_k': 3},
            confidence_scores=[0.7],
            raw_outputs=None
        )
        
        aggregated = model_manager.aggregate_predictions([result1, result2], method='hard_voting')
        
        assert aggregated.model_info['aggregation_method'] == 'hard_voting'
        assert len(aggregated.predictions) >= 1
        assert aggregated.predictions[0].species_name == "species_a"
    
    def test_performance_stats_update(self, model_manager):
        """Test performance statistics update."""
        mock_model = Mock()
        mock_model.config.model_name = "test_model"
        
        mock_result = Mock(spec=PredictionResult)
        mock_result.processing_time = 0.5
        mock_result.predictions = [Mock(confidence=0.8)]
        
        model_manager._update_performance_stats(mock_model, mock_result, True)
        
        stats = model_manager.model_performance["test_model"]
        assert stats['total_predictions'] == 1
        assert stats['reliable_predictions'] == 1
        assert stats['avg_processing_time'] == 0.5
        assert stats['avg_confidence'] == 0.8
    
    def test_get_performance_summary(self, model_manager):
        """Test performance summary generation."""
        model_manager.model_performance = {
            "test_model": {
                'total_predictions': 10,
                'reliable_predictions': 8,
                'avg_processing_time': 0.3,
                'avg_confidence': 0.75
            }
        }
        
        summary = model_manager.get_performance_summary()
        
        assert "test_model" in summary
        assert summary["test_model"]['reliability_rate'] == 0.8
        assert summary["test_model"]['avg_processing_time'] == 0.3
    
    def test_get_model_info(self, model_manager):
        """Test model info retrieval."""
        mock_primary = Mock()
        mock_primary.get_model_info.return_value = {'name': 'primary'}
        
        mock_fallback = Mock()
        mock_fallback.get_model_info.return_value = {'name': 'fallback'}
        
        model_manager.primary_model = mock_primary
        model_manager.fallback_models = [mock_fallback]
        
        info = model_manager.get_model_info()
        
        assert info['primary_model']['name'] == 'primary'
        assert len(info['fallback_models']) == 1
        assert info['fallback_models'][0]['name'] == 'fallback'
        assert 'performance_stats' in info
