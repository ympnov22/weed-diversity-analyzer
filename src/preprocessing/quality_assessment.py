"""Advanced image quality assessment for preprocessing pipeline."""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

from ..utils.logger import LoggerMixin


@dataclass
class QualityThresholds:
    """Quality assessment thresholds."""
    
    min_blur_score: float = 100.0
    min_exposure_score: float = 0.5
    max_noise_level: float = 0.3
    min_contrast_score: float = 20.0
    min_sharpness_score: float = 50.0


class QualityAssessor(LoggerMixin):
    """Advanced image quality assessment for Swin Transformer compatibility."""
    
    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        """Initialize quality assessor.
        
        Args:
            thresholds: Quality thresholds. If None, uses defaults.
        """
        self.thresholds = thresholds or QualityThresholds()
    
    def assess_blur(self, image: np.ndarray[Any, np.dtype[Any]]) -> float:
        """Assess image blur using Laplacian variance.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Blur score (higher = sharper)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(laplacian_var)
    
    def assess_exposure(self, image: np.ndarray[Any, np.dtype[Any]]) -> Dict[str, float]:
        """Assess image exposure characteristics.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary with exposure metrics
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        mean_brightness = float(np.mean(gray))
        std_brightness = float(np.std(gray))
        
        underexposed = np.sum(gray < 25) / gray.size
        overexposed = np.sum(gray > 230) / gray.size
        
        exposure_score = 1.0 - (underexposed + overexposed)
        
        dynamic_range = np.max(gray) - np.min(gray)
        
        return {
            'mean_brightness': float(mean_brightness),
            'std_brightness': float(std_brightness),
            'underexposed_ratio': float(underexposed),
            'overexposed_ratio': float(overexposed),
            'exposure_score': float(exposure_score),
            'dynamic_range': float(dynamic_range)
        }
    
    def assess_noise(self, image: np.ndarray[Any, np.dtype[Any]]) -> float:
        """Assess image noise level.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Noise level (0-1, lower is better)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        median_filtered = cv2.medianBlur(gray, 5)
        noise = np.abs(gray.astype(np.float32) - median_filtered.astype(np.float32))
        noise_level = np.mean(noise) / 255.0
        
        return float(noise_level)
    
    def assess_contrast(self, image: np.ndarray[Any, np.dtype[Any]]) -> float:
        """Assess image contrast using standard deviation.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Contrast score (higher = more contrast)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        contrast_score = float(np.std(gray))
        return contrast_score
    
    def assess_sharpness(self, image: np.ndarray[Any, np.dtype[Any]]) -> float:
        """Assess image sharpness using gradient magnitude.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Sharpness score (higher = sharper)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        sharpness_score = np.mean(gradient_magnitude)
        
        return float(sharpness_score)
    
    def assess_swin_compatibility(self, image: np.ndarray[Any, np.dtype[Any]]) -> Dict[str, float]:
        """Assess image compatibility with Swin Transformer requirements.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary with Swin Transformer compatibility metrics
        """
        h, w = image.shape[:2]
        
        aspect_ratio = w / h
        aspect_ratio_score = 1.0 - abs(aspect_ratio - 1.0)  # Prefer square images
        
        min_resolution = min(h, w)
        resolution_score = min(1.0, min_resolution / 224.0)  # Swin Transformer input size
        
        if len(image.shape) == 3:
            mean_b, mean_g, mean_r = np.mean(image, axis=(0, 1))
            color_balance = 1.0 - (np.std([mean_b, mean_g, mean_r]) / 255.0)
        else:
            color_balance = 0.5  # Grayscale penalty
        
        patches = self._extract_patches(image, patch_size=32)
        patch_diversity = self._calculate_patch_diversity(patches)
        
        return {
            'aspect_ratio_score': float(aspect_ratio_score),
            'resolution_score': float(resolution_score),
            'color_balance_score': float(color_balance),
            'patch_diversity_score': float(patch_diversity),
            'overall_swin_score': float(np.mean([
                aspect_ratio_score, resolution_score, color_balance, patch_diversity
            ]))
        }
    
    def _extract_patches(self, image: np.ndarray[Any, np.dtype[Any]], patch_size: int = 32) -> np.ndarray[Any, np.dtype[Any]]:
        """Extract patches from image for diversity analysis.
        
        Args:
            image: Input image
            patch_size: Size of patches to extract
            
        Returns:
            Array of patches
        """
        h, w = image.shape[:2]
        patches = []
        
        for y in range(0, h - patch_size + 1, patch_size):
            for x in range(0, w - patch_size + 1, patch_size):
                patch = image[y:y+patch_size, x:x+patch_size]
                if len(patch.shape) == 3:
                    patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                else:
                    patch_gray = patch
                patches.append(patch_gray.flatten())
        
        return np.array(patches)
    
    def _calculate_patch_diversity(self, patches: np.ndarray[Any, np.dtype[Any]]) -> float:
        """Calculate diversity among image patches.
        
        Args:
            patches: Array of flattened patches
            
        Returns:
            Diversity score (0-1, higher = more diverse)
        """
        if len(patches) < 2:
            return 0.0
        
        correlations = []
        for i in range(len(patches)):
            for j in range(i + 1, len(patches)):
                corr = np.corrcoef(patches[i], patches[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        if not correlations:
            return 0.0
        
        diversity_score = 1.0 - float(np.mean(correlations))
        return max(0.0, diversity_score)
    
    def comprehensive_assessment(self, image: np.ndarray[Any, np.dtype[Any]]) -> Dict[str, Any]:
        """Perform comprehensive quality assessment.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary with all quality metrics
        """
        blur_score = self.assess_blur(image)
        exposure_metrics = self.assess_exposure(image)
        noise_level = self.assess_noise(image)
        contrast_score = self.assess_contrast(image)
        sharpness_score = self.assess_sharpness(image)
        swin_metrics = self.assess_swin_compatibility(image)
        
        quality_components = [
            min(1.0, blur_score / 100.0),  # Normalize blur score
            exposure_metrics['exposure_score'],
            1.0 - noise_level,  # Invert noise (lower is better)
            min(1.0, contrast_score / 50.0),  # Normalize contrast
            min(1.0, sharpness_score / 100.0),  # Normalize sharpness
            swin_metrics['overall_swin_score']
        ]
        
        overall_quality = np.mean(quality_components)
        
        is_acceptable = (
            blur_score >= self.thresholds.min_blur_score and
            exposure_metrics['exposure_score'] >= self.thresholds.min_exposure_score and
            noise_level <= self.thresholds.max_noise_level and
            contrast_score >= self.thresholds.min_contrast_score and
            sharpness_score >= self.thresholds.min_sharpness_score
        )
        
        return {
            'blur_score': blur_score,
            'exposure_metrics': exposure_metrics,
            'noise_level': noise_level,
            'contrast_score': contrast_score,
            'sharpness_score': sharpness_score,
            'swin_compatibility': swin_metrics,
            'overall_quality_score': float(overall_quality),
            'is_acceptable': is_acceptable,
            'quality_components': {
                'blur': min(1.0, blur_score / 100.0),
                'exposure': exposure_metrics['exposure_score'],
                'noise': 1.0 - noise_level,
                'contrast': min(1.0, contrast_score / 50.0),
                'sharpness': min(1.0, sharpness_score / 100.0),
                'swin_compatibility': swin_metrics['overall_swin_score']
            }
        }
    
    def assess_image_file(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """Assess quality of image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Quality assessment results or None if image cannot be loaded
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.warning(f"Cannot load image: {image_path}")
                return None
            
            assessment = self.comprehensive_assessment(image)
            assessment['image_path'] = str(image_path)
            assessment['image_size'] = image.shape[:2]
            
            self.logger.debug(f"Quality assessment for {image_path}: {assessment['overall_quality_score']:.3f}")
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error assessing image {image_path}: {e}")
            return None
