"""Tests for JSON exporter implementation."""

import pytest
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import shutil

from src.output.json_exporter import JSONExporter
from src.utils.data_structures import DiversityMetrics, PredictionResult, SpeciesPrediction


class TestJSONExporter:
    """Test cases for JSON exporter."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def json_exporter(self, temp_output_dir):
        """Create JSON exporter instance."""
        return JSONExporter(temp_output_dir / "json")
    
    @pytest.fixture
    def sample_diversity_metrics(self):
        """Create sample diversity metrics."""
        from datetime import datetime
        return DiversityMetrics(
            date=datetime(2025, 8, 24),
            total_images=10,
            processed_images=8,
            species_richness=5,
            shannon_diversity=1.5,
            pielou_evenness=0.8,
            simpson_diversity=0.7,
            hill_q0=5.0,
            hill_q1=4.5,
            hill_q2=3.8,
            chao1_estimate=6.2,
            coverage_estimate=0.85
        )
    
    @pytest.fixture
    def sample_prediction_results(self):
        """Create sample prediction results."""
        from datetime import datetime
        from pathlib import Path
        
        predictions1 = [
            SpeciesPrediction("Species A", 0.9),
            SpeciesPrediction("Species B", 0.7),
            SpeciesPrediction("Species C", 0.5)
        ]
        
        predictions2 = [
            SpeciesPrediction("Species A", 0.8),
            SpeciesPrediction("Species D", 0.6),
            SpeciesPrediction("Species E", 0.4)
        ]
        
        return [
            PredictionResult(
                image_path=Path("image_001.jpg"),
                predictions=predictions1,
                model_name="inatag",
                processing_time=0.1,
                timestamp=datetime(2025, 8, 24, 12, 0, 0)
            ),
            PredictionResult(
                image_path=Path("image_002.jpg"),
                predictions=predictions2,
                model_name="inatag",
                processing_time=0.15,
                timestamp=datetime(2025, 8, 24, 12, 1, 0)
            )
        ]
    
    @pytest.fixture
    def sample_processing_metadata(self):
        """Create sample processing metadata."""
        return {
            "model_info": {
                "name": "inatag",
                "size": "base",
                "lora_enabled": False
            },
            "quality_stats": {
                "total_images": 10,
                "accepted_images": 8,
                "rejection_rate": 0.2
            },
            "clustering_stats": {
                "clusters_formed": 3,
                "representative_images": 2
            },
            "processing_stats": {
                "total_time": 2.5,
                "average_time_per_image": 0.125
            }
        }
    
    def test_initialization(self, temp_output_dir):
        """Test JSON exporter initialization."""
        exporter = JSONExporter(temp_output_dir / "json")
        
        assert exporter.output_dir == temp_output_dir / "json"
        assert exporter.output_dir.exists()
    
    def test_initialization_default_dir(self):
        """Test JSON exporter with default directory."""
        exporter = JSONExporter()
        
        assert exporter.output_dir == Path("output/json")
    
    def test_export_daily_summary_success(
        self, 
        json_exporter, 
        sample_diversity_metrics, 
        sample_prediction_results,
        sample_processing_metadata
    ):
        """Test successful daily summary export."""
        date_str = "2025-08-24"
        confidence_intervals = {
            "shannon_diversity": (1.2, 1.8),
            "species_richness": (4.0, 6.0)
        }
        
        result = json_exporter.export_daily_summary(
            date_str=date_str,
            diversity_metrics=sample_diversity_metrics,
            prediction_results=sample_prediction_results,
            processing_metadata=sample_processing_metadata,
            confidence_intervals=confidence_intervals
        )
        
        assert result["date"] == date_str
        assert "timestamp" in result
        assert "diversity_metrics" in result
        assert "confidence_intervals" in result
        assert "top_species" in result
        assert "processing_info" in result
        assert "model_info" in result
        assert "metadata" in result
        
        diversity_metrics = result["diversity_metrics"]
        assert diversity_metrics["species_richness"] == 5
        assert diversity_metrics["shannon_diversity"] == 1.5
        
        assert result["confidence_intervals"] == confidence_intervals
        
        top_species = result["top_species"]
        assert len(top_species) > 0
        assert top_species[0]["species_name"] == "Species A"
        assert top_species[0]["count"] == 2  # Appears in both results
        
        output_file = json_exporter.output_dir / f"daily_summary_{date_str}.json"
        assert output_file.exists()
        
        with open(output_file, 'r') as f:
            loaded_data = json.load(f)
            assert loaded_data["date"] == date_str
    
    def test_export_daily_summary_empty_predictions(
        self, 
        json_exporter, 
        sample_diversity_metrics,
        sample_processing_metadata
    ):
        """Test daily summary export with empty predictions."""
        date_str = "2025-08-24"
        
        result = json_exporter.export_daily_summary(
            date_str=date_str,
            diversity_metrics=sample_diversity_metrics,
            prediction_results=[],
            processing_metadata=sample_processing_metadata
        )
        
        assert result["date"] == date_str
        assert result["top_species"] == []
        assert result["metadata"]["total_images_processed"] == 0
    
    def test_export_time_series_data(self, json_exporter):
        """Test time series data export."""
        daily_summaries = [
            {
                "date": "2025-08-24",
                "diversity_metrics": {
                    "species_richness": 5,
                    "shannon_diversity": 1.5,
                    "pielou_evenness": 0.8,
                    "simpson_diversity": 0.7
                },
                "top_species": [
                    {"species_name": "Species A", "count": 3}
                ],
                "processing_info": {"total_processing_time": 2.5},
                "metadata": {"total_images_processed": 10}
            },
            {
                "date": "2025-08-25",
                "diversity_metrics": {
                    "species_richness": 6,
                    "shannon_diversity": 1.7,
                    "pielou_evenness": 0.85,
                    "simpson_diversity": 0.75
                },
                "top_species": [
                    {"species_name": "Species B", "count": 4}
                ],
                "processing_info": {"total_processing_time": 3.0},
                "metadata": {"total_images_processed": 12}
            }
        ]
        
        result = json_exporter.export_time_series_data(
            daily_summaries=daily_summaries,
            start_date="2025-08-24",
            end_date="2025-08-25"
        )
        
        assert result["period"]["start_date"] == "2025-08-24"
        assert result["period"]["end_date"] == "2025-08-25"
        assert result["period"]["total_days"] == 2
        
        diversity_trends = result["diversity_trends"]
        assert len(diversity_trends["dates"]) == 2
        assert diversity_trends["species_richness"] == [5, 6]
        assert diversity_trends["shannon_diversity"] == [1.5, 1.7]
        
        species_trends = result["species_trends"]
        assert species_trends["total_unique_species"] == 2
        assert "Species A" in species_trends["species_list"]
        assert "Species B" in species_trends["species_list"]
        
        output_file = json_exporter.output_dir / "time_series_2025-08-24_to_2025-08-25.json"
        assert output_file.exists()
    
    def test_export_github_calendar_data(self, json_exporter):
        """Test GitHub calendar data export."""
        daily_summaries = [
            {
                "date": "2025-08-24",
                "diversity_metrics": {
                    "shannon_diversity": 1.5,
                    "species_richness": 5
                },
                "metadata": {"total_images_processed": 10}
            },
            {
                "date": "2025-08-25",
                "diversity_metrics": {
                    "shannon_diversity": 2.0,
                    "species_richness": 7
                },
                "metadata": {"total_images_processed": 15}
            }
        ]
        
        result = json_exporter.export_github_calendar_data(
            daily_summaries=daily_summaries,
            metric_name="shannon_diversity"
        )
        
        assert result["metric"] == "shannon_diversity"
        assert len(result["data"]) == 2
        assert "scale" in result
        assert "metadata" in result
        
        first_entry = result["data"][0]
        assert first_entry["date"] == "2025-08-24"
        assert first_entry["value"] == 1.5
        assert first_entry["species_count"] == 5
        assert first_entry["total_images"] == 10
        assert "level" in first_entry
        
        scale = result["scale"]
        assert "min" in scale
        assert "max" in scale
        assert "q25" in scale
        assert "q50" in scale
        assert "q75" in scale
        
        output_file = json_exporter.output_dir / "github_calendar_shannon_diversity.json"
        assert output_file.exists()
    
    def test_extract_top_species(self, json_exporter, sample_prediction_results):
        """Test top species extraction."""
        top_species = json_exporter._extract_top_species(sample_prediction_results, top_k=3)
        
        assert len(top_species) == 3  # Species A, B, C (D and E have count 1 each)
        
        assert top_species[0]["species_name"] == "Species A"
        assert top_species[0]["count"] == 2
        assert top_species[0]["frequency"] == 1.0  # 2/2 results
        
        assert 0.8 <= top_species[0]["average_confidence"] <= 0.9
    
    def test_calculate_processing_stats(self, json_exporter, sample_prediction_results, sample_processing_metadata):
        """Test processing statistics calculation."""
        stats = json_exporter._calculate_processing_stats(
            sample_prediction_results, 
            sample_processing_metadata
        )
        
        assert "total_processing_time" in stats
        assert "average_processing_time" in stats
        assert "min_processing_time" in stats
        assert "max_processing_time" in stats
        assert "average_confidence" in stats
        assert "total_predictions" in stats
        assert "successful_predictions" in stats
        
        assert stats["total_predictions"] == 2
        assert stats["successful_predictions"] == 2
        assert stats["total_processing_time"] == 0.25  # 0.1 + 0.15
    
    def test_serialize_diversity_metrics(self, json_exporter, sample_diversity_metrics):
        """Test diversity metrics serialization."""
        serialized = json_exporter._serialize_diversity_metrics(sample_diversity_metrics)
        
        assert isinstance(serialized, dict)
        assert serialized["species_richness"] == 5
        assert serialized["shannon_diversity"] == 1.5
        assert serialized["pielou_evenness"] == 0.8
    
    def test_serialize_diversity_metrics_dict(self, json_exporter):
        """Test diversity metrics serialization from dict."""
        metrics_dict = {
            "species_richness": np.int64(5),
            "shannon_diversity": np.float64(1.5),
            "pielou_evenness": 0.8
        }
        
        serialized = json_exporter._serialize_diversity_metrics(metrics_dict)
        
        assert isinstance(serialized, dict)
        assert isinstance(serialized["species_richness"], (int, float))
        assert isinstance(serialized["shannon_diversity"], (int, float))
    
    def test_calculate_intensity_level(self, json_exporter):
        """Test intensity level calculation for GitHub calendar."""
        scale = {"q25": 1.0, "q50": 2.0, "q75": 3.0}
        
        assert json_exporter._calculate_intensity_level(0.5, scale) == 1
        assert json_exporter._calculate_intensity_level(1.5, scale) == 2
        assert json_exporter._calculate_intensity_level(2.5, scale) == 3
        assert json_exporter._calculate_intensity_level(4.0, scale) == 4
    
    def test_json_serializer(self, json_exporter):
        """Test custom JSON serializer."""
        assert json_exporter._json_serializer(np.int64(5)) == 5
        assert json_exporter._json_serializer(np.float64(1.5)) == 1.5
        assert json_exporter._json_serializer(np.array([1, 2, 3])) == [1, 2, 3]
        
        dt = datetime(2025, 8, 24, 12, 0, 0)
        assert json_exporter._json_serializer(dt) == "2025-08-24T12:00:00"
        
        with pytest.raises(TypeError):
            json_exporter._json_serializer(object())
    
    def test_export_error_handling(self, json_exporter):
        """Test error handling in export methods."""
        try:
            result = json_exporter.export_daily_summary(
                date_str="2025-08-24",
                diversity_metrics=None,
                prediction_results=[],
                processing_metadata={}
            )
            assert result is not None
        except Exception:
            pass
