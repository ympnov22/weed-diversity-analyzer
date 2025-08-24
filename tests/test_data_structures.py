"""Tests for data structures."""

import pytest
from datetime import datetime
from pathlib import Path
from src.utils.data_structures import (
    ImageData, SpeciesPrediction, PredictionResult, DiversityMetrics
)


class TestImageData:
    """Test cases for ImageData."""
    
    def test_image_data_creation(self):
        """Test creating ImageData instance."""
        image_data = ImageData(
            path=Path("test.jpg"),
            date=datetime.now(),
            size=(1024, 768),
            file_size_bytes=1024000
        )
        
        assert isinstance(image_data.path, Path)
        assert isinstance(image_data.date, datetime)
        assert image_data.size == (1024, 768)
        assert image_data.file_size_bytes == 1024000
        assert not image_data.is_processed
        assert not image_data.is_representative
    
    def test_string_path_conversion(self):
        """Test automatic conversion of string path to Path object."""
        image_data = ImageData(
            path="test.jpg",
            date=datetime.now(),
            size=(1024, 768),
            file_size_bytes=1024000
        )
        
        assert isinstance(image_data.path, Path)
        assert str(image_data.path) == "test.jpg"


class TestSpeciesPrediction:
    """Test cases for SpeciesPrediction."""
    
    def test_valid_prediction(self):
        """Test creating valid species prediction."""
        prediction = SpeciesPrediction(
            species_name="Taraxacum officinale",
            confidence=0.85,
            taxonomic_level="species",
            scientific_name="Taraxacum officinale",
            common_name="Common dandelion"
        )
        
        assert prediction.species_name == "Taraxacum officinale"
        assert prediction.confidence == 0.85
        assert prediction.taxonomic_level == "species"
    
    def test_invalid_confidence(self):
        """Test validation of confidence values."""
        with pytest.raises(ValueError):
            SpeciesPrediction(
                species_name="Test species",
                confidence=1.5  # Invalid: > 1.0
            )
        
        with pytest.raises(ValueError):
            SpeciesPrediction(
                species_name="Test species",
                confidence=-0.1  # Invalid: < 0.0
            )


class TestPredictionResult:
    """Test cases for PredictionResult."""
    
    def test_prediction_result_creation(self):
        """Test creating PredictionResult with calculations."""
        predictions = [
            SpeciesPrediction("Species A", 0.8),
            SpeciesPrediction("Species B", 0.6),
            SpeciesPrediction("Species C", 0.4),
        ]
        
        result = PredictionResult(
            image_path=Path("test.jpg"),
            predictions=predictions,
            model_name="test_model",
            processing_time=1.5
        )
        
        assert result.top_prediction.species_name == "Species A"
        assert result.top_prediction.confidence == 0.8
        assert abs(result.mean_confidence - 0.6) < 0.001
        assert result.prediction_entropy > 0  # Should have some entropy
    
    def test_get_top_k(self):
        """Test getting top-k predictions."""
        predictions = [
            SpeciesPrediction("Species A", 0.8),
            SpeciesPrediction("Species B", 0.6),
            SpeciesPrediction("Species C", 0.4),
            SpeciesPrediction("Species D", 0.2),
        ]
        
        result = PredictionResult(
            image_path=Path("test.jpg"),
            predictions=predictions,
            model_name="test_model",
            processing_time=1.5
        )
        
        top_2 = result.get_top_k(2)
        assert len(top_2) == 2
        assert top_2[0].species_name == "Species A"
        assert top_2[1].species_name == "Species B"
    
    def test_confident_prediction(self):
        """Test confidence threshold checking."""
        predictions = [
            SpeciesPrediction("Species A", 0.8),
            SpeciesPrediction("Species B", 0.3),
        ]
        
        result = PredictionResult(
            image_path=Path("test.jpg"),
            predictions=predictions,
            model_name="test_model",
            processing_time=1.5
        )
        
        assert result.has_confident_prediction(0.5)  # 0.8 > 0.5
        assert not result.has_confident_prediction(0.9)  # 0.8 < 0.9


class TestDiversityMetrics:
    """Test cases for DiversityMetrics."""
    
    def test_diversity_metrics_creation(self):
        """Test creating DiversityMetrics instance."""
        species_counts = {
            "Species A": 10,
            "Species B": 5,
            "Species C": 3,
        }
        
        metrics = DiversityMetrics(
            date=datetime.now(),
            total_images=100,
            processed_images=95,
            species_richness=3,
            shannon_diversity=1.5,
            pielou_evenness=0.8,
            simpson_diversity=0.7,
            hill_q0=3.0,
            hill_q1=4.5,
            hill_q2=3.3,
            species_counts=species_counts
        )
        
        assert metrics.species_richness == 3
        assert metrics.shannon_diversity == 1.5
        assert len(metrics.species_frequencies) == 3
        assert abs(metrics.species_frequencies["Species A"] - 10/18) < 0.001
    
    def test_to_dict_serialization(self):
        """Test dictionary serialization."""
        metrics = DiversityMetrics(
            date=datetime(2025, 8, 24),
            total_images=100,
            processed_images=95,
            species_richness=3,
            shannon_diversity=1.5,
            pielou_evenness=0.8,
            simpson_diversity=0.7,
            hill_q0=3.0,
            hill_q1=4.5,
            hill_q2=3.3,
        )
        
        data_dict = metrics.to_dict()
        assert isinstance(data_dict, dict)
        assert data_dict["date"] == "2025-08-24T00:00:00"
        assert data_dict["species_richness"] == 3
        assert "hill_numbers" in data_dict
        assert data_dict["hill_numbers"]["q0"] == 3.0
