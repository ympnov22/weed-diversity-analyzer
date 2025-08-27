"""Multi-model management system with fallback and aggregation."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time

from .base_model import BaseModel, PredictionResult
from .inatag_model import iNatAgModel
from ..utils.logger import LoggerMixin
from ..utils.config import ConfigManager


class ModelManager(LoggerMixin):
    """Manages multiple models with fallback and result aggregation."""
    
    def __init__(self, config: ConfigManager):
        """Initialize model manager.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.primary_model = None
        self.fallback_models = []
        self.model_performance = {}
        self.prediction_cache = {}
        
    def load_models(self) -> bool:
        """Load primary and fallback models."""
        try:
            primary_config = self.config.get_primary_model()
            if not primary_config:
                self.logger.error("No primary model configuration found")
                return False
            
            self.primary_model = self._create_model_from_config(primary_config)
            if not self.primary_model.load_model():
                self.logger.error("Failed to load primary model")
                return False
            
            fallback_configs = self.config.get_fallback_models()
            for i, fallback_config in enumerate(fallback_configs):
                fallback_model = self._create_model_from_config(fallback_config)
                if fallback_model.load_model():
                    self.fallback_models.append(fallback_model)
                    self.logger.info(f"Loaded fallback model {i+1}")
                else:
                    self.logger.warning(f"Failed to load fallback model {i+1}")
            
            self.logger.info(f"Loaded {1 + len(self.fallback_models)} models total")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            return False
    
    def _create_model_from_config(self, model_config) -> BaseModel:
        """Create model instance from configuration.
        
        Args:
            model_config: ModelConfig instance from ConfigManager
            
        Returns:
            Model instance
        """
        from .base_model import ModelConfig
        
        config = ModelConfig(
            model_name=model_config.model_name,
            model_path=model_config.model_path,
            confidence_threshold=model_config.confidence_threshold,
            top_k=model_config.top_k or 3,
            device='auto',
            batch_size=32,
            use_lora=False,
            lora_path=None
        )
        
        model_size = 'base'  # Default size
        if hasattr(model_config, 'size'):
            model_size = model_config.size
        elif 'tiny' in model_config.model_name.lower():
            model_size = 'tiny'
        elif 'small' in model_config.model_name.lower():
            model_size = 'small'
        elif 'large' in model_config.model_name.lower():
            model_size = 'large'
        
        return iNatAgModel(config, model_size)
    
    def predict_with_fallback(self, image: np.ndarray) -> PredictionResult:
        """Predict with primary model and fallback if needed.
        
        Args:
            image: Input image
            
        Returns:
            Prediction result
        """
        if self.primary_model is None:
            raise RuntimeError("No models loaded")
        
        result = self.primary_model.predict(image)
        
        if self.primary_model.is_prediction_reliable(result):
            self._update_performance_stats(self.primary_model, result, True)
            return result
        
        self.logger.info("Primary model prediction unreliable, trying fallback models")
        
        for i, fallback_model in enumerate(self.fallback_models):
            fallback_result = fallback_model.predict(image)
            if fallback_model.is_prediction_reliable(fallback_result):
                self.logger.info(f"Used fallback model {i+1} for prediction")
                self._update_performance_stats(fallback_model, fallback_result, True)
                return fallback_result
        
        self.logger.warning("No reliable predictions from any model")
        self._update_performance_stats(self.primary_model, result, False)
        return result
    
    def predict_batch_with_fallback(self, images: List[np.ndarray]) -> List[PredictionResult]:
        """Predict batch with fallback strategy.
        
        Args:
            images: List of input images
            
        Returns:
            List of prediction results
        """
        if self.primary_model is None:
            raise RuntimeError("No models loaded")
        
        results = self.primary_model.predict_batch(images)
        
        unreliable_indices = []
        for i, result in enumerate(results):
            if not self.primary_model.is_prediction_reliable(result):
                unreliable_indices.append(i)
        
        if unreliable_indices:
            self.logger.info(f"Retrying {len(unreliable_indices)} unreliable predictions with fallback")
            
            for fallback_model in self.fallback_models:
                if not unreliable_indices:
                    break
                
                unreliable_images = [images[i] for i in unreliable_indices]
                fallback_results = fallback_model.predict_batch(unreliable_images)
                
                new_unreliable = []
                for j, (idx, fallback_result) in enumerate(zip(unreliable_indices, fallback_results)):
                    if fallback_model.is_prediction_reliable(fallback_result):
                        results[idx] = fallback_result
                    else:
                        new_unreliable.append(idx)
                
                unreliable_indices = new_unreliable
        
        return results
    
    def aggregate_predictions(self, prediction_results: List[PredictionResult], method: str = 'soft_voting') -> PredictionResult:
        """Aggregate predictions from multiple models.
        
        Args:
            prediction_results: List of prediction results to aggregate
            method: Aggregation method ('soft_voting', 'hard_voting', 'confidence_weighted')
            
        Returns:
            Aggregated prediction result
        """
        if not prediction_results:
            raise ValueError("No prediction results to aggregate")
        
        if len(prediction_results) == 1:
            return prediction_results[0]
        
        if method == 'soft_voting':
            return self._soft_voting_aggregation(prediction_results)
        elif method == 'hard_voting':
            return self._hard_voting_aggregation(prediction_results)
        elif method == 'confidence_weighted':
            return self._confidence_weighted_aggregation(prediction_results)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def _soft_voting_aggregation(self, results: List[PredictionResult]) -> PredictionResult:
        """Aggregate using soft voting (probability averaging)."""
        if not all(r.raw_outputs is not None for r in results):
            return self._hard_voting_aggregation(results)
        
        num_classes = len(results[0].raw_outputs)
        aggregated_probs = np.zeros(num_classes)
        
        for result in results:
            aggregated_probs += result.raw_outputs
        
        aggregated_probs /= len(results)
        
        from .base_model import ModelConfig
        dummy_config = ModelConfig(model_name="aggregated", model_path="", top_k=results[0].model_info.get('top_k', 3))
        dummy_model = iNatAgModel(dummy_config)
        dummy_model.species_list = results[0].predictions[0].species_name if results[0].predictions else []
        
        predictions = dummy_model.postprocess_predictions(aggregated_probs)
        
        return PredictionResult(
            predictions=predictions,
            processing_time=sum(r.processing_time for r in results),
            model_info={'aggregation_method': 'soft_voting', 'num_models': len(results)},
            confidence_scores=[p.confidence for p in predictions],
            raw_outputs=aggregated_probs
        )
    
    def _hard_voting_aggregation(self, results: List[PredictionResult]) -> PredictionResult:
        """Aggregate using hard voting (majority vote)."""
        species_votes = {}
        
        for result in results:
            for prediction in result.predictions:
                species_name = prediction.species_name
                if species_name not in species_votes:
                    species_votes[species_name] = []
                species_votes[species_name].append(prediction.confidence)
        
        aggregated_predictions = []
        for species_name, confidences in species_votes.items():
            avg_confidence = np.mean(confidences)
            vote_count = len(confidences)
            
            from ..utils.data_structures import SpeciesPrediction
            prediction = SpeciesPrediction(
                species_name=species_name,
                confidence=avg_confidence * (vote_count / len(results)),
                scientific_name=species_name,
                common_name=""
            )
            aggregated_predictions.append(prediction)
        
        aggregated_predictions.sort(key=lambda x: x.confidence, reverse=True)
        top_k = results[0].model_info.get('top_k', 3)
        aggregated_predictions = aggregated_predictions[:top_k]
        
        return PredictionResult(
            predictions=aggregated_predictions,
            processing_time=sum(r.processing_time for r in results),
            model_info={'aggregation_method': 'hard_voting', 'num_models': len(results)},
            confidence_scores=[p.confidence for p in aggregated_predictions],
            raw_outputs=None
        )
    
    def _confidence_weighted_aggregation(self, results: List[PredictionResult]) -> PredictionResult:
        """Aggregate using confidence-weighted averaging."""
        if not all(r.raw_outputs is not None for r in results):
            return self._hard_voting_aggregation(results)
        
        num_classes = len(results[0].raw_outputs)
        weighted_probs = np.zeros(num_classes)
        total_weight = 0.0
        
        for result in results:
            if result.predictions:
                weight = result.predictions[0].confidence
                weighted_probs += result.raw_outputs * weight
                total_weight += weight
        
        if total_weight > 0:
            weighted_probs /= total_weight
        
        from .base_model import ModelConfig
        dummy_config = ModelConfig(model_name="aggregated", model_path="", top_k=results[0].model_info.get('top_k', 3))
        dummy_model = iNatAgModel(dummy_config)
        dummy_model.species_list = [f"species_{i}" for i in range(num_classes)]
        
        predictions = dummy_model.postprocess_predictions(weighted_probs)
        
        return PredictionResult(
            predictions=predictions,
            processing_time=sum(r.processing_time for r in results),
            model_info={'aggregation_method': 'confidence_weighted', 'num_models': len(results)},
            confidence_scores=[p.confidence for p in predictions],
            raw_outputs=weighted_probs
        )
    
    def _update_performance_stats(self, model: BaseModel, result: PredictionResult, reliable: bool):
        """Update model performance statistics."""
        model_name = model.config.model_name
        
        if model_name not in self.model_performance:
            self.model_performance[model_name] = {
                'total_predictions': 0,
                'reliable_predictions': 0,
                'avg_processing_time': 0.0,
                'avg_confidence': 0.0
            }
        
        stats = self.model_performance[model_name]
        stats['total_predictions'] += 1
        
        if reliable:
            stats['reliable_predictions'] += 1
        
        stats['avg_processing_time'] = (
            (stats['avg_processing_time'] * (stats['total_predictions'] - 1) + result.processing_time) /
            stats['total_predictions']
        )
        
        if result.predictions:
            top_confidence = result.predictions[0].confidence
            stats['avg_confidence'] = (
                (stats['avg_confidence'] * (stats['total_predictions'] - 1) + top_confidence) /
                stats['total_predictions']
            )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all models."""
        summary = {}
        
        for model_name, stats in self.model_performance.items():
            reliability_rate = (
                stats['reliable_predictions'] / stats['total_predictions']
                if stats['total_predictions'] > 0 else 0.0
            )
            
            summary[model_name] = {
                'total_predictions': stats['total_predictions'],
                'reliability_rate': reliability_rate,
                'avg_processing_time': stats['avg_processing_time'],
                'avg_confidence': stats['avg_confidence']
            }
        
        return summary
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about all loaded models."""
        info = {
            'primary_model': self.primary_model.get_model_info() if self.primary_model else None,
            'fallback_models': [model.get_model_info() for model in self.fallback_models],
            'performance_stats': self.get_performance_summary()
        }
        
        return info
