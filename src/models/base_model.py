"""Base model interface for plant species classification."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
# import numpy as np  # Removed for minimal deployment
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
    raw_outputs: Optional[Any] = None  # np.ndarray removed for minimal deployment


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
    def predict(self, image: Any) -> PredictionResult:  # np.ndarray removed for minimal deployment
        """Predict species from image.
        
        Args:
            image: Preprocessed image (normalized, resized)
            
        Returns:
            Prediction result with top-k species
        """
        pass
    
    @abstractmethod
    def predict_batch(self, images: List[Any]) -> List[PredictionResult]:  # np.ndarray removed for minimal deployment
        """Predict species for batch of images.
        
        Args:
            images: List of preprocessed images
            
        Returns:
            List of prediction results
        """
        pass
    
    def preprocess_image(self, image: Any) -> Any:  # np.ndarray removed for minimal deployment
        """Preprocess image for model input.
        
        Args:
            image: Raw image (BGR format, 0-255 range)
            
        Returns:
            Preprocessed image ready for model
        """
        self.logger.warning("Image preprocessing not available in minimal mode")
        return image
    
    def postprocess_predictions(self, raw_outputs: Any) -> List[SpeciesPrediction]:  # np.ndarray removed for minimal deployment
        """Convert raw model outputs to species predictions.
        
        Args:
            raw_outputs: Raw model outputs (logits or probabilities)
            
        Returns:
            List of species predictions
        """
        self.logger.warning("Prediction postprocessing not available in minimal mode")
        
        if self.species_list:
            prediction = SpeciesPrediction(
                species_name=self.species_list[0],
                confidence=0.5,
                scientific_name=self.species_list[0],
                common_name=""
            )
            return [prediction]
        
        return []
    
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
