"""Complete inference pipeline integrating preprocessing and models."""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time
import cv2

from .preprocessing.image_processor import ImageProcessor, PreprocessingConfig
from .preprocessing.similarity_clustering import SimilarityClusterer, ClusteringConfig
from .preprocessing.quality_assessment import QualityAssessor, QualityThresholds
from .models.model_manager import ModelManager
from .utils.config import ConfigManager
from .utils.logger import LoggerMixin
from .utils.data_structures import ImageData, SpeciesPrediction


class InferencePipeline(LoggerMixin):
    """Complete inference pipeline from raw images to species predictions."""
    
    def __init__(self, config: ConfigManager):
        """Initialize inference pipeline.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        
        preprocessing_config = PreprocessingConfig()
        clustering_config = ClusteringConfig()
        quality_thresholds = QualityThresholds()
        
        self.image_processor = ImageProcessor(preprocessing_config)
        self.clusterer = SimilarityClusterer(clustering_config)
        self.quality_assessor = QualityAssessor(quality_thresholds)
        self.model_manager = ModelManager(config)
        
        if not self.model_manager.load_models():
            raise RuntimeError("Failed to load models")
    
    def process_daily_images(self, image_dir: Path) -> Dict[str, Any]:
        """Process all images from a daily folder.
        
        Args:
            image_dir: Directory containing daily images
            
        Returns:
            Dictionary with processing results and predictions
        """
        start_time = time.time()
        
        if not image_dir.exists():
            raise ValueError(f"Image directory does not exist: {image_dir}")
        
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpeg"))
        
        if not image_files:
            self.logger.warning(f"No image files found in {image_dir}")
            return self._empty_result(image_dir.name, time.time() - start_time)
        
        self.logger.info(f"Processing {len(image_files)} images from {image_dir}")
        
        quality_results = self._assess_image_quality(image_files)
        if not quality_results:
            self.logger.warning("No images passed quality assessment")
            return self._empty_result(image_dir.name, time.time() - start_time)
        
        representative_images = self._cluster_and_select_representatives(quality_results)
        
        predictions = self._run_inference(representative_images)
        
        processing_time = time.time() - start_time
        
        return {
            'date': image_dir.name,
            'total_images': len(image_files),
            'accepted_images': len(quality_results),
            'representative_images': len(representative_images),
            'predictions': predictions,
            'processing_time': processing_time,
            'quality_summary': self._summarize_quality_results(quality_results),
            'clustering_summary': self._summarize_clustering(quality_results, representative_images)
        }
    
    def _assess_image_quality(self, image_files: List[Path]) -> List[Tuple[Path, Dict[str, Any]]]:
        """Assess quality of all images."""
        quality_results = []
        
        for image_file in image_files:
            quality_result = self.quality_assessor.assess_image_file(image_file)
            if quality_result and quality_result['is_acceptable']:
                quality_results.append((image_file, quality_result))
            else:
                self.logger.debug(f"Image {image_file.name} failed quality assessment")
        
        self.logger.info(f"Accepted {len(quality_results)}/{len(image_files)} images after quality assessment")
        return quality_results
    
    def _cluster_and_select_representatives(self, quality_results: List[Tuple[Path, Dict[str, Any]]]) -> List[Tuple[Path, np.ndarray]]:
        """Cluster similar images and select representatives."""
        if len(quality_results) <= 1:
            if quality_results:
                image_file, _ = quality_results[0]
                image = cv2.imread(str(image_file))
                if image is not None:
                    return [(image_file, image)]
            return []
        
        images_for_clustering = []
        for image_file, quality_data in quality_results:
            image = cv2.imread(str(image_file))
            if image is not None:
                images_for_clustering.append((image_file, image))
        
        if not images_for_clustering:
            return []
        
        try:
            representative_images = self.clusterer.process_image_batch(images_for_clustering)
            self.logger.info(f"Selected {len(representative_images)} representative images from {len(images_for_clustering)} candidates")
            return representative_images
        except Exception as e:
            self.logger.error(f"Clustering failed: {e}, using all images")
            return images_for_clustering
    
    def _run_inference(self, representative_images: List[Tuple[Path, np.ndarray]]) -> List[Dict[str, Any]]:
        """Run inference on representative images."""
        predictions = []
        
        for image_file, image in representative_images:
            try:
                processed_image, metadata = self.image_processor.process_image(image_file)
                if processed_image is not None:
                    prediction_result = self.model_manager.predict_with_fallback(processed_image)
                    
                    predictions.append({
                        'image_file': str(image_file),
                        'image_name': image_file.name,
                        'predictions': [
                            {
                                'species_name': pred.species_name,
                                'confidence': pred.confidence,
                                'scientific_name': pred.scientific_name,
                                'common_name': pred.common_name,
                                'family': pred.family,
                                'genus': pred.genus
                            }
                            for pred in prediction_result.predictions
                        ],
                        'processing_time': prediction_result.processing_time,
                        'model_info': prediction_result.model_info,
                        'preprocessing_metadata': metadata
                    })
                else:
                    self.logger.warning(f"Failed to preprocess image: {image_file}")
                    
            except Exception as e:
                self.logger.error(f"Inference failed for {image_file}: {e}")
        
        self.logger.info(f"Completed inference on {len(predictions)} images")
        return predictions
    
    def _empty_result(self, date: str, processing_time: float) -> Dict[str, Any]:
        """Create empty result structure."""
        return {
            'date': date,
            'total_images': 0,
            'accepted_images': 0,
            'representative_images': 0,
            'predictions': [],
            'processing_time': processing_time,
            'quality_summary': {},
            'clustering_summary': {}
        }
    
    def _summarize_quality_results(self, quality_results: List[Tuple[Path, Dict[str, Any]]]) -> Dict[str, Any]:
        """Summarize quality assessment results."""
        if not quality_results:
            return {}
        
        quality_scores = [result[1]['overall_quality_score'] for result in quality_results]
        swin_scores = [result[1]['swin_compatibility']['overall_swin_score'] for result in quality_results]
        
        return {
            'avg_quality_score': float(np.mean(quality_scores)),
            'min_quality_score': float(np.min(quality_scores)),
            'max_quality_score': float(np.max(quality_scores)),
            'avg_swin_compatibility': float(np.mean(swin_scores)),
            'quality_distribution': {
                'excellent': sum(1 for score in quality_scores if score >= 0.8),
                'good': sum(1 for score in quality_scores if 0.6 <= score < 0.8),
                'acceptable': sum(1 for score in quality_scores if 0.4 <= score < 0.6),
                'poor': sum(1 for score in quality_scores if score < 0.4)
            }
        }
    
    def _summarize_clustering(self, quality_results: List[Tuple[Path, Dict[str, Any]]], representatives: List[Tuple[Path, np.ndarray]]) -> Dict[str, Any]:
        """Summarize clustering results."""
        return {
            'input_images': len(quality_results),
            'representative_images': len(representatives),
            'reduction_ratio': len(representatives) / len(quality_results) if quality_results else 0.0,
            'clustering_method': self.clusterer.config.clustering_method,
            'similarity_method': self.clusterer.config.similarity_method
        }
    
    def process_single_image(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """Process a single image through the complete pipeline.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Prediction result or None if processing failed
        """
        try:
            quality_result = self.quality_assessor.assess_image_file(image_path)
            if not quality_result or not quality_result['is_acceptable']:
                self.logger.warning(f"Image {image_path} failed quality assessment")
                return None
            
            processed_image, metadata = self.image_processor.process_image(image_path)
            if processed_image is None:
                self.logger.warning(f"Failed to preprocess image: {image_path}")
                return None
            
            prediction_result = self.model_manager.predict_with_fallback(processed_image)
            
            return {
                'image_file': str(image_path),
                'image_name': image_path.name,
                'predictions': [
                    {
                        'species_name': pred.species_name,
                        'confidence': pred.confidence,
                        'scientific_name': pred.scientific_name,
                        'common_name': pred.common_name,
                        'family': pred.family,
                        'genus': pred.genus
                    }
                    for pred in prediction_result.predictions
                ],
                'processing_time': prediction_result.processing_time,
                'model_info': prediction_result.model_info,
                'quality_assessment': quality_result,
                'preprocessing_metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process single image {image_path}: {e}")
            return None
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline configuration."""
        return {
            'preprocessing': {
                'target_size': self.image_processor.config.target_size,
                'enhancement_enabled': self.image_processor.config.enable_clahe,
                'white_balance_method': self.image_processor.config.white_balance_method
            },
            'quality_assessment': {
                'min_blur_score': self.quality_assessor.thresholds.min_blur_score,
                'min_exposure_score': self.quality_assessor.thresholds.min_exposure_score,
                'max_noise_level': self.quality_assessor.thresholds.max_noise_level
            },
            'clustering': {
                'similarity_method': self.clusterer.config.similarity_method,
                'clustering_method': self.clusterer.config.clustering_method,
                'similarity_threshold': self.clusterer.config.similarity_threshold
            },
            'models': self.model_manager.get_model_info()
        }
