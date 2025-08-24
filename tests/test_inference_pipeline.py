"""Tests for inference pipeline implementation."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import cv2

from src.inference_pipeline import InferencePipeline
from src.utils.config import ConfigManager


class TestInferencePipeline:
    """Test cases for inference pipeline."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration manager."""
        config = Mock(spec=ConfigManager)
        return config
    
    @pytest.fixture
    def mock_components(self):
        """Create mock pipeline components."""
        mock_image_processor = Mock()
        mock_clusterer = Mock()
        mock_quality_assessor = Mock()
        mock_model_manager = Mock()
        mock_model_manager.load_models.return_value = True
        
        return {
            'image_processor': mock_image_processor,
            'clusterer': mock_clusterer,
            'quality_assessor': mock_quality_assessor,
            'model_manager': mock_model_manager
        }
    
    @pytest.fixture
    def inference_pipeline(self, mock_config, mock_components):
        """Create inference pipeline with mocked components."""
        with patch('src.inference_pipeline.ImageProcessor', return_value=mock_components['image_processor']), \
             patch('src.inference_pipeline.SimilarityClusterer', return_value=mock_components['clusterer']), \
             patch('src.inference_pipeline.QualityAssessor', return_value=mock_components['quality_assessor']), \
             patch('src.inference_pipeline.ModelManager', return_value=mock_components['model_manager']):
            
            pipeline = InferencePipeline(mock_config)
            
        return pipeline
    
    def test_initialization_success(self, mock_config, mock_components):
        """Test successful pipeline initialization."""
        with patch('src.inference_pipeline.ImageProcessor', return_value=mock_components['image_processor']), \
             patch('src.inference_pipeline.SimilarityClusterer', return_value=mock_components['clusterer']), \
             patch('src.inference_pipeline.QualityAssessor', return_value=mock_components['quality_assessor']), \
             patch('src.inference_pipeline.ModelManager', return_value=mock_components['model_manager']):
            
            pipeline = InferencePipeline(mock_config)
            
            assert pipeline.image_processor == mock_components['image_processor']
            assert pipeline.clusterer == mock_components['clusterer']
            assert pipeline.quality_assessor == mock_components['quality_assessor']
            assert pipeline.model_manager == mock_components['model_manager']
    
    def test_initialization_model_load_failure(self, mock_config):
        """Test pipeline initialization with model loading failure."""
        mock_model_manager = Mock()
        mock_model_manager.load_models.return_value = False
        
        with patch('src.inference_pipeline.ImageProcessor'), \
             patch('src.inference_pipeline.SimilarityClusterer'), \
             patch('src.inference_pipeline.QualityAssessor'), \
             patch('src.inference_pipeline.ModelManager', return_value=mock_model_manager):
            
            with pytest.raises(RuntimeError, match="Failed to load models"):
                InferencePipeline(mock_config)
    
    def test_process_daily_images_nonexistent_directory(self, inference_pipeline):
        """Test processing with nonexistent directory."""
        nonexistent_dir = Path("/nonexistent/directory")
        
        with pytest.raises(ValueError, match="Image directory does not exist"):
            inference_pipeline.process_daily_images(nonexistent_dir)
    
    def test_process_daily_images_no_images(self, inference_pipeline):
        """Test processing directory with no images."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir)
            
            result = inference_pipeline.process_daily_images(image_dir)
            
            assert result['total_images'] == 0
            assert result['accepted_images'] == 0
            assert result['representative_images'] == 0
            assert result['predictions'] == []
    
    def test_assess_image_quality(self, inference_pipeline):
        """Test image quality assessment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir)
            
            test_image_path = image_dir / "test.jpg"
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            cv2.imwrite(str(test_image_path), test_image)
            
            inference_pipeline.quality_assessor.assess_image_file.return_value = {
                'is_acceptable': True,
                'overall_quality_score': 0.8
            }
            
            image_files = [test_image_path]
            quality_results = inference_pipeline._assess_image_quality(image_files)
            
            assert len(quality_results) == 1
            assert quality_results[0][0] == test_image_path
    
    def test_cluster_and_select_representatives_single_image(self, inference_pipeline):
        """Test clustering with single image."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "test.jpg"
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            cv2.imwrite(str(image_path), test_image)
            
            quality_results = [(image_path, {'is_acceptable': True})]
            
            representatives = inference_pipeline._cluster_and_select_representatives(quality_results)
            
            assert len(representatives) == 1
            assert representatives[0][0] == image_path
    
    def test_cluster_and_select_representatives_multiple_images(self, inference_pipeline):
        """Test clustering with multiple images."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_paths = []
            quality_results = []
            
            for i in range(3):
                image_path = Path(temp_dir) / f"test_{i}.jpg"
                test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                cv2.imwrite(str(image_path), test_image)
                
                image_paths.append(image_path)
                quality_results.append((image_path, {'is_acceptable': True}))
            
            mock_representatives = [(image_paths[0], np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))]
            inference_pipeline.clusterer.process_image_batch.return_value = mock_representatives
            
            representatives = inference_pipeline._cluster_and_select_representatives(quality_results)
            
            assert len(representatives) == 1
            inference_pipeline.clusterer.process_image_batch.assert_called_once()
    
    def test_run_inference(self, inference_pipeline):
        """Test inference execution."""
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image_path = Path("test.jpg")
        
        representative_images = [(image_path, test_image)]
        
        inference_pipeline.image_processor.process_image.return_value = (test_image, {'processed': True})
        
        mock_prediction_result = Mock()
        mock_prediction_result.predictions = [
            Mock(species_name="test_species", confidence=0.8, scientific_name="Test species",
                 common_name="Test", family="Testaceae", genus="Testus")
        ]
        mock_prediction_result.processing_time = 0.1
        mock_prediction_result.model_info = {'model': 'test'}
        
        inference_pipeline.model_manager.predict_with_fallback.return_value = mock_prediction_result
        
        predictions = inference_pipeline._run_inference(representative_images)
        
        assert len(predictions) == 1
        assert predictions[0]['image_name'] == 'test.jpg'
        assert len(predictions[0]['predictions']) == 1
        assert predictions[0]['predictions'][0]['species_name'] == 'test_species'
    
    def test_process_single_image_success(self, inference_pipeline):
        """Test successful single image processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "test.jpg"
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            cv2.imwrite(str(image_path), test_image)
            
            inference_pipeline.quality_assessor.assess_image_file.return_value = {
                'is_acceptable': True,
                'overall_quality_score': 0.8
            }
            
            inference_pipeline.image_processor.process_image.return_value = (test_image, {'processed': True})
            
            mock_prediction_result = Mock()
            mock_prediction_result.predictions = [
                Mock(species_name="test_species", confidence=0.8, scientific_name="Test species",
                     common_name="Test", family="Testaceae", genus="Testus")
            ]
            mock_prediction_result.processing_time = 0.1
            mock_prediction_result.model_info = {'model': 'test'}
            
            inference_pipeline.model_manager.predict_with_fallback.return_value = mock_prediction_result
            
            result = inference_pipeline.process_single_image(image_path)
            
            assert result is not None
            assert result['image_name'] == 'test.jpg'
            assert len(result['predictions']) == 1
    
    def test_process_single_image_quality_failure(self, inference_pipeline):
        """Test single image processing with quality failure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "test.jpg"
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            cv2.imwrite(str(image_path), test_image)
            
            inference_pipeline.quality_assessor.assess_image_file.return_value = {
                'is_acceptable': False,
                'overall_quality_score': 0.2
            }
            
            result = inference_pipeline.process_single_image(image_path)
            
            assert result is None
    
    def test_summarize_quality_results(self, inference_pipeline):
        """Test quality results summarization."""
        quality_results = [
            (Path("test1.jpg"), {
                'overall_quality_score': 0.8,
                'swin_compatibility': {'overall_swin_score': 0.9}
            }),
            (Path("test2.jpg"), {
                'overall_quality_score': 0.6,
                'swin_compatibility': {'overall_swin_score': 0.7}
            })
        ]
        
        summary = inference_pipeline._summarize_quality_results(quality_results)
        
        assert summary['avg_quality_score'] == 0.7
        assert summary['min_quality_score'] == 0.6
        assert summary['max_quality_score'] == 0.8
        assert summary['avg_swin_compatibility'] == 0.8
        assert 'quality_distribution' in summary
    
    def test_summarize_clustering(self, inference_pipeline):
        """Test clustering results summarization."""
        quality_results = [(Path(f"test{i}.jpg"), {}) for i in range(5)]
        representatives = [(Path("test0.jpg"), np.array([]))]
        
        inference_pipeline.clusterer.config.clustering_method = "hierarchical"
        inference_pipeline.clusterer.config.similarity_method = "ssim"
        
        summary = inference_pipeline._summarize_clustering(quality_results, representatives)
        
        assert summary['input_images'] == 5
        assert summary['representative_images'] == 1
        assert summary['reduction_ratio'] == 0.2
        assert summary['clustering_method'] == "hierarchical"
        assert summary['similarity_method'] == "ssim"
    
    def test_get_pipeline_info(self, inference_pipeline):
        """Test pipeline information retrieval."""
        inference_pipeline.image_processor.config.target_size = (224, 224)
        inference_pipeline.image_processor.config.enable_clahe = True
        inference_pipeline.image_processor.config.white_balance_method = "gray_world"
        
        inference_pipeline.quality_assessor.thresholds.min_blur_score = 100.0
        inference_pipeline.quality_assessor.thresholds.min_exposure_score = 0.5
        inference_pipeline.quality_assessor.thresholds.max_noise_level = 0.3
        
        inference_pipeline.clusterer.config.similarity_method = "ssim"
        inference_pipeline.clusterer.config.clustering_method = "hierarchical"
        inference_pipeline.clusterer.config.similarity_threshold = 0.8
        
        inference_pipeline.model_manager.get_model_info.return_value = {'models': 'info'}
        
        info = inference_pipeline.get_pipeline_info()
        
        assert info['preprocessing']['target_size'] == (224, 224)
        assert info['quality_assessment']['min_blur_score'] == 100.0
        assert info['clustering']['similarity_method'] == "ssim"
        assert info['models'] == {'models': 'info'}
