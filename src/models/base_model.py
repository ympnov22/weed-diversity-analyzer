"""Base model interface for plant species classification."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from dataclasses import dataclass

from ..utils.logger import LoggerMixin
from ..utils.data_structures import SpeciesPrediction


@dataclass
class ModelConfig:
    """Configuration for model inference."""
    
    model_name: str
    model_path: str
    confidence_threshold: float = 0.5
    top_k: int = 3
    device: str = "auto"  # auto, cpu, cuda
    batch_size: int = 32
    
    input_size: Tuple[int, int] = (224, 224)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    use_lora: bool = False
    lora_path: Optional[str] = None
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1


@dataclass
class PredictionResult:
    """Result from model prediction."""
    
    predictions: List[SpeciesPrediction]
    processing_time: float
    model_info: Dict[str, Any]
    confidence_scores: List[float]
    raw_outputs: Optional[np.ndarray] = None


class BaseModel(ABC, LoggerMixin):
    """Abstract base class for plant species classification models."""
    
    def __init__(self, config: ModelConfig):
        """Initialize base model.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.model = None
        self.species_list: List[str] = []
        self.is_loaded = False
    
    @abstractmethod
    def load_model(self) -> bool:
        """Load the model from specified path.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> PredictionResult:
        """Predict species from image.
        
        Args:
            image: Preprocessed image (normalized, resized)
            
        Returns:
            Prediction result with top-k species
        """
        pass
    
    @abstractmethod
    def predict_batch(self, images: List[np.ndarray]) -> List[PredictionResult]:
        """Predict species for batch of images.
        
        Args:
            images: List of preprocessed images
            
        Returns:
            List of prediction results
        """
        pass
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input.
        
        Args:
            image: Raw image (BGR format, 0-255 range)
            
        Returns:
            Preprocessed image ready for model
        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = image[:, :, ::-1]  # BGR to RGB
        else:
            image_rgb = image
        
        if image_rgb.max() > 1.0:
            image_rgb = image_rgb.astype(np.float32) / 255.0
        
        mean = np.array(self.config.normalize_mean)
        std = np.array(self.config.normalize_std)
        
        normalized = (image_rgb - mean) / std
        
        return normalized
    
    def postprocess_predictions(self, raw_outputs: np.ndarray) -> List[SpeciesPrediction]:
        """Convert raw model outputs to species predictions.
        
        Args:
            raw_outputs: Raw model outputs (logits or probabilities)
            
        Returns:
            List of species predictions
        """
        if raw_outputs.max() > 1.0 or raw_outputs.min() < 0.0:
            exp_outputs = np.exp(raw_outputs - np.max(raw_outputs))
            probabilities = exp_outputs / np.sum(exp_outputs)
        else:
            probabilities = raw_outputs
        
        top_k_indices = np.argsort(probabilities)[-self.config.top_k:][::-1]
        
        predictions = []
        for idx in top_k_indices:
            if idx < len(self.species_list):
                species_name = self.species_list[idx]
                confidence = float(probabilities[idx])
                
                prediction = SpeciesPrediction(
                    species_name=species_name,
                    confidence=confidence,
                    scientific_name=species_name,
                    common_name=""
                )
                predictions.append(prediction)
        
        return predictions
    
    def is_prediction_reliable(self, prediction_result: PredictionResult) -> bool:
        """Check if prediction is reliable based on confidence threshold.
        
        Args:
            prediction_result: Prediction result to evaluate
            
        Returns:
            True if prediction is reliable
        """
        if not prediction_result.predictions:
            return False
        
        top_confidence = prediction_result.predictions[0].confidence
        return top_confidence >= self.config.confidence_threshold
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            'model_name': self.config.model_name,
            'model_path': self.config.model_path,
            'num_species': len(self.species_list),
            'input_size': self.config.input_size,
            'confidence_threshold': self.config.confidence_threshold,
            'top_k': self.config.top_k,
            'use_lora': self.config.use_lora,
            'is_loaded': self.is_loaded
        }
    
    def load_species_list(self, species_file: Optional[Path] = None) -> bool:
        """Load species list from file.
        
        Args:
            species_file: Path to species list file. If None, uses default.
            
        Returns:
            True if species list loaded successfully
        """
        try:
            if species_file is None:
                model_dir = Path(self.config.model_path).parent
                species_file = model_dir / "species_list.txt"
            
            if not species_file.exists():
                self.logger.warning(f"Species file not found: {species_file}")
                return False
            
            with open(species_file, 'r', encoding='utf-8') as f:
                self.species_list = [line.strip() for line in f if line.strip()]
            
            self.logger.info(f"Loaded {len(self.species_list)} species from {species_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading species list: {e}")
            return False
    
    def validate_model(self) -> bool:
        """Validate that model is properly loaded and configured.
        
        Returns:
            True if model is valid
        """
        if not self.is_loaded:
            self.logger.error("Model not loaded")
            return False
        
        if not self.species_list:
            self.logger.error("Species list not loaded")
            return False
        
        if self.model is None:
            self.logger.error("Model object is None")
            return False
        
        return True
    
    def __str__(self) -> str:
        """String representation of model."""
        return f"{self.__class__.__name__}({self.config.model_name}, {len(self.species_list)} species)"
