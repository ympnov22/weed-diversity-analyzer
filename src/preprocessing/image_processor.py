"""Image preprocessing module for quality enhancement and normalization."""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from ..utils.logger import LoggerMixin
from ..utils.data_structures import ImageData


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing."""
    
    target_size: Tuple[int, int] = (224, 224)
    
    clahe_enabled: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    
    white_balance_enabled: bool = True
    white_balance_method: str = "gray_world"  # gray_world, white_patch
    
    gamma_correction_enabled: bool = True
    gamma_range: Tuple[float, float] = (0.8, 1.2)
    
    min_resolution: Tuple[int, int] = (512, 512)
    max_file_size_mb: float = 50.0


class ImageProcessor(LoggerMixin):
    """Handles image preprocessing including quality enhancement and normalization."""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize image processor.
        
        Args:
            config: Preprocessing configuration. If None, uses defaults.
        """
        self.config = config or PreprocessingConfig()
        self.clahe = None
        
        if self.config.clahe_enabled:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=self.config.clahe_tile_grid_size
            )
    
    def load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """Load image from file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Loaded image as numpy array (BGR format) or None if failed
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.warning(f"Failed to load image: {image_path}")
                return None
            
            self.logger.debug(f"Loaded image {image_path} with shape {image.shape}")
            return image
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def assess_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Assess image quality metrics.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary of quality metrics
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        brightness_mean = np.mean(gray)
        
        contrast_std = np.std(gray)
        
        underexposed = np.sum(gray < 25) / gray.size
        overexposed = np.sum(gray > 230) / gray.size
        exposure_score = 1.0 - (underexposed + overexposed)
        
        return {
            'blur_score': float(blur_score),
            'brightness_mean': float(brightness_mean),
            'contrast_std': float(contrast_std),
            'exposure_score': float(exposure_score),
            'underexposed_ratio': float(underexposed),
            'overexposed_ratio': float(overexposed)
        }
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            CLAHE-enhanced image
        """
        if not self.config.clahe_enabled or self.clahe is None:
            return image
        
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        self.logger.debug("Applied CLAHE enhancement")
        return enhanced
    
    def apply_white_balance(self, image: np.ndarray) -> np.ndarray:
        """Apply white balance correction.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            White-balanced image
        """
        if not self.config.white_balance_enabled:
            return image
        
        if self.config.white_balance_method == "gray_world":
            return self._gray_world_white_balance(image)
        elif self.config.white_balance_method == "white_patch":
            return self._white_patch_white_balance(image)
        else:
            self.logger.warning(f"Unknown white balance method: {self.config.white_balance_method}")
            return image
    
    def _gray_world_white_balance(self, image: np.ndarray) -> np.ndarray:
        """Apply gray world white balance assumption.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            White-balanced image
        """
        mean_b = np.mean(image[:, :, 0])
        mean_g = np.mean(image[:, :, 1])
        mean_r = np.mean(image[:, :, 2])
        
        overall_mean = (mean_b + mean_g + mean_r) / 3
        
        scale_b = overall_mean / mean_b if mean_b > 0 else 1.0
        scale_g = overall_mean / mean_g if mean_g > 0 else 1.0
        scale_r = overall_mean / mean_r if mean_r > 0 else 1.0
        
        balanced = image.astype(np.float32)
        balanced[:, :, 0] *= scale_b
        balanced[:, :, 1] *= scale_g
        balanced[:, :, 2] *= scale_r
        
        balanced = np.clip(balanced, 0, 255).astype(np.uint8)
        
        self.logger.debug(f"Applied gray world white balance (scales: B={scale_b:.3f}, G={scale_g:.3f}, R={scale_r:.3f})")
        return balanced
    
    def _white_patch_white_balance(self, image: np.ndarray) -> np.ndarray:
        """Apply white patch white balance assumption.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            White-balanced image
        """
        max_b = np.max(image[:, :, 0])
        max_g = np.max(image[:, :, 1])
        max_r = np.max(image[:, :, 2])
        
        scale_b = 255.0 / max_b if max_b > 0 else 1.0
        scale_g = 255.0 / max_g if max_g > 0 else 1.0
        scale_r = 255.0 / max_r if max_r > 0 else 1.0
        
        balanced = image.astype(np.float32)
        balanced[:, :, 0] *= scale_b
        balanced[:, :, 1] *= scale_g
        balanced[:, :, 2] *= scale_r
        
        balanced = np.clip(balanced, 0, 255).astype(np.uint8)
        
        self.logger.debug(f"Applied white patch white balance (scales: B={scale_b:.3f}, G={scale_g:.3f}, R={scale_r:.3f})")
        return balanced
    
    def apply_gamma_correction(self, image: np.ndarray, gamma: Optional[float] = None) -> np.ndarray:
        """Apply gamma correction for exposure adjustment.
        
        Args:
            image: Input image (BGR format)
            gamma: Gamma value. If None, automatically determined.
            
        Returns:
            Gamma-corrected image
        """
        if not self.config.gamma_correction_enabled:
            return image
        
        if gamma is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray) / 255.0
            
            if mean_brightness < 0.3:  # Dark image
                gamma = np.random.uniform(0.8, 1.0)
            elif mean_brightness > 0.7:  # Bright image
                gamma = np.random.uniform(1.0, 1.2)
            else:  # Normal brightness
                gamma = np.random.uniform(*self.config.gamma_range)
        
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        
        corrected = cv2.LUT(image, table)
        
        self.logger.debug(f"Applied gamma correction (gamma={gamma:.3f})")
        return corrected
    
    def resize_image(self, image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Resize image to target size.
        
        Args:
            image: Input image
            target_size: Target size (width, height). If None, uses config default.
            
        Returns:
            Resized image
        """
        if target_size is None:
            target_size = self.config.target_size
        
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
        self.logger.debug(f"Resized image to {target_size}")
        return resized
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range.
        
        Args:
            image: Input image (0-255 range)
            
        Returns:
            Normalized image (0-1 range)
        """
        normalized = image.astype(np.float32) / 255.0
        return normalized
    
    def process_image(self, image_path: Path, save_intermediate: bool = False) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Complete image preprocessing pipeline.
        
        Args:
            image_path: Path to input image
            save_intermediate: Whether to save intermediate processing steps
            
        Returns:
            Tuple of (processed_image, metadata) or None if processing failed
        """
        image = self.load_image(image_path)
        if image is None:
            return None
        
        h, w = image.shape[:2]
        if w < self.config.min_resolution[0] or h < self.config.min_resolution[1]:
            self.logger.warning(f"Image {image_path} below minimum resolution: {w}x{h}")
            return None
        
        quality_metrics = self.assess_image_quality(image)
        
        processed = image.copy()
        
        processed = self.apply_clahe(processed)
        
        processed = self.apply_white_balance(processed)
        
        processed = self.apply_gamma_correction(processed)
        
        processed = self.resize_image(processed)
        
        processed = self.normalize_image(processed)
        
        metadata = {
            'original_size': (w, h),
            'processed_size': self.config.target_size,
            'quality_metrics': quality_metrics,
            'processing_steps': [
                'clahe' if self.config.clahe_enabled else None,
                'white_balance' if self.config.white_balance_enabled else None,
                'gamma_correction' if self.config.gamma_correction_enabled else None,
                'resize',
                'normalize'
            ]
        }
        
        metadata['processing_steps'] = [step for step in metadata['processing_steps'] if step is not None]
        
        self.logger.info(f"Successfully processed image {image_path}")
        return processed, metadata
    
    def is_image_acceptable(self, image_path: Path) -> Tuple[bool, str]:
        """Check if image meets quality requirements.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (is_acceptable, reason)
        """
        try:
            file_size_mb = image_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.config.max_file_size_mb:
                return False, f"File size too large: {file_size_mb:.1f}MB > {self.config.max_file_size_mb}MB"
        except Exception as e:
            return False, f"Cannot access file: {e}"
        
        image = self.load_image(image_path)
        if image is None:
            return False, "Cannot load image"
        
        h, w = image.shape[:2]
        if w < self.config.min_resolution[0] or h < self.config.min_resolution[1]:
            return False, f"Resolution too low: {w}x{h} < {self.config.min_resolution}"
        
        quality = self.assess_image_quality(image)
        
        if quality['blur_score'] < 100:
            return False, f"Image too blurry: {quality['blur_score']:.1f}"
        
        if quality['exposure_score'] < 0.5:
            return False, f"Poor exposure: {quality['exposure_score']:.3f}"
        
        return True, "Image acceptable"
