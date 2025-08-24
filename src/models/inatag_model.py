"""iNatAg Swin Transformer model implementation."""

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
import timm
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time

from .base_model import BaseModel, ModelConfig, PredictionResult
from ..utils.data_structures import SpeciesPrediction


class iNatAgModel(BaseModel):
    """iNatAg Swin Transformer model for plant species classification."""
    
    AVAILABLE_SIZES = {
        'tiny': {'params': '28M', 'file_size': '117MB'},
        'small': {'params': '50M', 'file_size': '203MB'}, 
        'base': {'params': '88M', 'file_size': '360MB'},
        'large': {'params': '197M', 'file_size': '798MB'}
    }
    
    def __init__(self, config: ModelConfig, model_size: str = 'base'):
        """Initialize iNatAg model.
        
        Args:
            config: Model configuration
            model_size: Model size ('tiny', 'small', 'base', 'large')
        """
        super().__init__(config)
        self.model_size = model_size
        self.repo_id = "Project-AgML/iNatAg-models"
        self.num_classes = 2959
        self.device = None
        
    def _get_device(self) -> torch.device:
        """Get appropriate device for model."""
        if self.device is not None:
            return self.device
            
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        
        self.device = device
        return device
    
    def _get_model_filename(self) -> str:
        """Get model filename based on size and LoRA configuration."""
        if self.config.use_lora:
            return f"swin_{self.model_size}_with_lora.pth"
        else:
            return f"swin_{self.model_size}_without_lora.pth"
    
    def load_model(self) -> bool:
        """Load iNatAg model from Hugging Face Hub."""
        try:
            model_filename = self._get_model_filename()
            
            self.logger.info(f"Downloading {model_filename} from {self.repo_id}")
            model_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=model_filename,
                cache_dir=Path(self.config.model_path).parent / "cache"
            )
            
            timm_model_name = f'swin_{self.model_size}_patch4_window7_224'
            self.model = timm.create_model(
                timm_model_name,
                pretrained=False,
                num_classes=self.num_classes
            )
            
            state_dict = torch.load(model_path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            
            self.model.load_state_dict(state_dict, strict=False)
            
            device = self._get_device()
            self.model = self.model.to(device)
            self.model.eval()
            
            self._load_inatag_species_list()
            
            self.is_loaded = True
            self.logger.info(f"Loaded iNatAg {self.model_size} model with {self.num_classes} species on {device}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load iNatAg model: {e}")
            return False
    
    def _load_inatag_species_list(self) -> bool:
        """Load iNatAg species list."""
        try:
            species_filename = "species_list.txt"
            try:
                species_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=species_filename,
                    cache_dir=Path(self.config.model_path).parent / "cache"
                )
                
                with open(species_path, 'r', encoding='utf-8') as f:
                    self.species_list = [line.strip() for line in f if line.strip()]
                    
            except Exception:
                self.logger.warning("Species list not found in repository, generating placeholder")
                self.species_list = [f"species_{i:04d}" for i in range(self.num_classes)]
            
            self.logger.info(f"Loaded {len(self.species_list)} species")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading species list: {e}")
            return False
    
    def predict(self, image: np.ndarray) -> PredictionResult:
        """Predict species from image."""
        if not self.validate_model():
            raise RuntimeError("Model not properly loaded")
        
        start_time = time.time()
        
        try:
            preprocessed = self.preprocess_image(image)
            
            if len(preprocessed.shape) == 3:
                preprocessed = np.expand_dims(preprocessed, axis=0)
            
            input_tensor = torch.from_numpy(preprocessed).float()
            input_tensor = input_tensor.to(self._get_device())
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                probabilities = torch.softmax(outputs, dim=1)
                raw_outputs = probabilities.cpu().numpy()[0]
            
            predictions = self.postprocess_predictions(raw_outputs)
            processing_time = time.time() - start_time
            
            return PredictionResult(
                predictions=predictions,
                processing_time=processing_time,
                model_info=self.get_model_info(),
                confidence_scores=[p.confidence for p in predictions],
                raw_outputs=raw_outputs
            )
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return PredictionResult(
                predictions=[],
                processing_time=time.time() - start_time,
                model_info=self.get_model_info(),
                confidence_scores=[],
                raw_outputs=None
            )
    
    def predict_batch(self, images: List[np.ndarray]) -> List[PredictionResult]:
        """Predict species for batch of images."""
        if not self.validate_model():
            raise RuntimeError("Model not properly loaded")
        
        results = []
        batch_size = min(self.config.batch_size, len(images))
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_results = self._predict_batch_internal(batch)
            results.extend(batch_results)
        
        return results
    
    def _predict_batch_internal(self, batch: List[np.ndarray]) -> List[PredictionResult]:
        """Internal batch prediction method."""
        start_time = time.time()
        
        try:
            preprocessed_batch = []
            for image in batch:
                preprocessed = self.preprocess_image(image)
                preprocessed_batch.append(preprocessed)
            
            batch_tensor = torch.from_numpy(np.stack(preprocessed_batch)).float()
            batch_tensor = batch_tensor.to(self._get_device())
            
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                probabilities = torch.softmax(outputs, dim=1)
                raw_outputs = probabilities.cpu().numpy()
            
            results = []
            processing_time = (time.time() - start_time) / len(batch)
            
            for i, raw_output in enumerate(raw_outputs):
                predictions = self.postprocess_predictions(raw_output)
                
                result = PredictionResult(
                    predictions=predictions,
                    processing_time=processing_time,
                    model_info=self.get_model_info(),
                    confidence_scores=[p.confidence for p in predictions],
                    raw_outputs=raw_output
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch prediction failed: {e}")
            return [PredictionResult(
                predictions=[],
                processing_time=0.0,
                model_info=self.get_model_info(),
                confidence_scores=[],
                raw_outputs=None
            ) for _ in batch]
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for Swin Transformer input."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = image[:, :, ::-1]
        else:
            image_rgb = image
        
        if image_rgb.max() > 1.0:
            image_rgb = image_rgb.astype(np.float32) / 255.0
        else:
            image_rgb = image_rgb.astype(np.float32)
        
        import cv2
        target_size = self.config.input_size
        if image_rgb.shape[:2] != target_size:
            image_rgb = cv2.resize(image_rgb, target_size)
        
        mean = np.array(self.config.normalize_mean, dtype=np.float32)
        std = np.array(self.config.normalize_std, dtype=np.float32)
        
        if len(image_rgb.shape) == 2:
            image_rgb = np.stack([image_rgb, image_rgb, image_rgb], axis=2)
        elif len(image_rgb.shape) == 3 and image_rgb.shape[2] == 1:
            image_rgb = np.repeat(image_rgb, 3, axis=2)
        
        if len(image_rgb.shape) == 3:
            normalized = (image_rgb - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
        else:
            normalized = (image_rgb - mean[0]) / std[0]
        
        if len(normalized.shape) == 3:
            normalized = normalized.transpose(2, 0, 1)
        
        return normalized.astype(np.float32)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        base_info = super().get_model_info()
        base_info.update({
            'model_size': self.model_size,
            'repo_id': self.repo_id,
            'architecture': 'Swin Transformer',
            'num_classes': self.num_classes,
            'device': str(self._get_device()) if self.device else 'not_loaded'
        })
        return base_info
