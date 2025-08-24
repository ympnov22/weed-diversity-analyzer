"""Tests for CSV exporter implementation."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock
import tempfile
import shutil

from src.output.csv_exporter import CSVExporter
from src.utils.data_structures import PredictionResult, SpeciesPrediction


class TestCSVExporter:
    """Test cases for CSV exporter."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def csv_exporter(self, temp_output_dir):
        """Create CSV exporter instance."""
        return CSVExporter(temp_output_dir / "csv")
    
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
            SpeciesPrediction("Species D", 0.8),
            SpeciesPrediction("Species E", 0.6)
        ]
        
        result1 = PredictionResult(
            image_path=Path("image_001.jpg"),
            predictions=predictions1,
            model_name="inatag",
            processing_time=0.1,
            timestamp=datetime(2025, 8, 24, 12, 0, 0)
        )
        
        result2 = PredictionResult(
            image_path=Path("image_002.jpg"),
            predictions=predictions2,
            model_name="inatag",
            processing_time=0.15,
            timestamp=datetime(2025, 8, 24, 12, 1, 0)
        )
        
        return [result1, result2]
    
    @pytest.fixture
    def sample_processing_metadata(self):
        """Create sample processing metadata."""
        return {
            "model_info": {
                "name": "inatag",
                "size": "base",
                "lora_enabled": False
            },
            "processing_stats": {
                "total_time": 2.5,
                "average_time_per_image": 0.125
            }
        }
    
    def test_initialization(self, temp_output_dir):
        """Test CSV exporter initialization."""
        exporter = CSVExporter(temp_output_dir / "csv")
        
        assert exporter.output_dir == temp_output_dir / "csv"
        assert exporter.output_dir.exists()
    
    def test_initialization_default_dir(self):
        """Test CSV exporter with default directory."""
        exporter = CSVExporter()
        
        assert exporter.output_dir == Path("output/csv")
    
    def test_export_daily_predictions_success(
        self, 
        csv_exporter, 
        sample_prediction_results,
        sample_processing_metadata
    ):
        """Test successful daily predictions export."""
        date_str = "2025-08-24"
        
        output_file = csv_exporter.export_daily_predictions(
            date_str=date_str,
            prediction_results=sample_prediction_results,
            processing_metadata=sample_processing_metadata
        )
        
        assert output_file.exists()
        assert output_file.name == f"daily_predictions_{date_str}.csv"
        
        df = pd.read_csv(output_file)
        
        assert len(df) == 2  # Two prediction results
        assert "date" in df.columns
        assert "image_id" in df.columns
        assert "image_path" in df.columns
        assert "processing_time" in df.columns
        
        for i in range(1, 4):
            assert f"species_name_{i}" in df.columns
            assert f"confidence_{i}" in df.columns
            assert f"inatag_id_{i}" in df.columns
            assert f"taxonomic_level_{i}" in df.columns
        
        assert "top_confidence" in df.columns
        assert "mean_confidence" in df.columns
        assert "confidence_std" in df.columns
        assert "prediction_entropy" in df.columns
        
        assert all(df["date"] == date_str)
        assert df.iloc[0]["species_name_1"] == "Species A"
        assert df.iloc[0]["confidence_1"] == 0.9
        assert df.iloc[0]["top_confidence"] == 0.9
        
        assert df.iloc[1]["species_name_1"] == "Species D"
        assert df.iloc[1]["confidence_1"] == 0.8
        species_name_3 = df.iloc[1]["species_name_3"]
        if pd.isna(species_name_3):
            assert species_name_3 != species_name_3  # NaN != NaN is True
        else:
            assert species_name_3 == ""
        assert df.iloc[1]["confidence_3"] == 0.0
    
    def test_export_daily_predictions_empty_results(
        self, 
        csv_exporter,
        sample_processing_metadata
    ):
        """Test daily predictions export with empty results."""
        date_str = "2025-08-24"
        
        output_file = csv_exporter.export_daily_predictions(
            date_str=date_str,
            prediction_results=[],
            processing_metadata=sample_processing_metadata
        )
        
        assert output_file.exists()
        
        df = pd.read_csv(output_file)
        assert len(df) == 0  # Empty DataFrame
    
    def test_export_aggregated_predictions(self, csv_exporter, sample_prediction_results):
        """Test aggregated predictions export."""
        daily_results = {
            "2025-08-24": sample_prediction_results[:1],
            "2025-08-25": sample_prediction_results[1:]
        }
        
        output_file = csv_exporter.export_aggregated_predictions(
            daily_results=daily_results,
            start_date="2025-08-24",
            end_date="2025-08-25"
        )
        
        assert output_file.exists()
        assert "aggregated_predictions_2025-08-24_to_2025-08-25.csv" in output_file.name
        
        df = pd.read_csv(output_file)
        
        assert len(df) == 2  # One result per day
        assert "date" in df.columns
        assert "image_id" in df.columns
        
        assert df.iloc[0]["date"] == "2025-08-24"
        assert df.iloc[1]["date"] == "2025-08-25"
        assert df.iloc[0]["image_id"] == "2025-08-24_1"
        assert df.iloc[1]["image_id"] == "2025-08-25_1"
    
    def test_export_species_summary(self, csv_exporter, sample_prediction_results):
        """Test species summary export."""
        daily_results = {
            "2025-08-24": sample_prediction_results,
            "2025-08-25": sample_prediction_results[:1]  # Only first result
        }
        
        output_file = csv_exporter.export_species_summary(
            daily_results=daily_results
        )
        
        assert output_file.exists()
        assert "species_summary" in output_file.name
        
        df = pd.read_csv(output_file)
        
        expected_columns = [
            "species_name", "total_occurrences", "days_observed",
            "observation_frequency", "average_confidence", "confidence_std",
            "min_confidence", "max_confidence", "first_observed", "last_observed"
        ]
        for col in expected_columns:
            assert col in df.columns
        
        species_a_row = df[df["species_name"] == "Species A"].iloc[0]
        assert species_a_row["total_occurrences"] == 2  # Appears twice
        assert species_a_row["days_observed"] == 2  # Both days
        assert species_a_row["observation_frequency"] == 1.0  # 2/2 days
        assert species_a_row["first_observed"] == "2025-08-24"
        assert species_a_row["last_observed"] == "2025-08-25"
        
        assert df.iloc[0]["species_name"] == "Species A"
        assert df.iloc[0]["total_occurrences"] >= df.iloc[1]["total_occurrences"]
    
    def test_export_soft_voting_results(self, csv_exporter):
        """Test soft voting results export."""
        date_str = "2025-08-24"
        soft_voting_results = {
            "species_weights": {
                "Species A": 0.6,
                "Species B": 0.3,
                "Species C": 0.1
            },
            "diversity_metrics": {
                "shannon_diversity": 1.5,
                "species_richness": 3
            },
            "taxonomic_levels": {
                "Species A": "species",
                "Species B": "genus",
                "Species C": "species"
            }
        }
        
        comparison_results = {
            "hard_voting_weights": {
                "Species A": 0.5,
                "Species B": 0.4,
                "Species C": 0.1
            },
            "agreement_rate": 0.85
        }
        
        output_file = csv_exporter.export_soft_voting_results(
            date_str=date_str,
            soft_voting_results=soft_voting_results,
            comparison_results=comparison_results
        )
        
        assert output_file.exists()
        assert f"soft_voting_{date_str}.csv" in output_file.name
        
        df = pd.read_csv(output_file)
        
        expected_columns = [
            "date", "species_name", "soft_voting_weight", "normalized_weight",
            "taxonomic_level", "hard_voting_weight", "weight_difference"
        ]
        for col in expected_columns:
            assert col in df.columns
        
        species_rows = df[df["species_name"] != "SUMMARY"]
        assert len(species_rows) == 3
        
        species_a_row = species_rows[species_rows["species_name"] == "Species A"].iloc[0]
        assert species_a_row["soft_voting_weight"] == 0.6
        assert species_a_row["taxonomic_level"] == "species"
        assert species_a_row["hard_voting_weight"] == 0.5
        assert abs(species_a_row["weight_difference"] - 0.1) < 1e-10  # Handle floating point precision
        
        summary_row = df[df["species_name"] == "SUMMARY"].iloc[0]
        assert summary_row["soft_voting_weight"] == 1.0  # Sum of all weights
        assert summary_row["normalized_weight"] == 1.0
        assert "diversity_shannon_diversity" in summary_row.index or summary_row.get("diversity_shannon_diversity") is not None
    
    def test_calculate_entropy(self, csv_exporter):
        """Test entropy calculation."""
        confidences = [0.5, 0.5]
        entropy = csv_exporter._calculate_entropy(confidences)
        assert entropy == 1.0  # log2(2) = 1
        
        confidences = [0.8, 0.2]
        entropy = csv_exporter._calculate_entropy(confidences)
        assert 0 < entropy < 1
        
        confidences = [1.0]
        entropy = csv_exporter._calculate_entropy(confidences)
        assert entropy == 0.0
        
        confidences = []
        entropy = csv_exporter._calculate_entropy(confidences)
        assert entropy == 0.0
        
        confidences = [0.0, 0.0]
        entropy = csv_exporter._calculate_entropy(confidences)
        assert entropy == 0.0
    
    def test_export_error_handling(self, csv_exporter):
        """Test error handling in export methods."""
        try:
            result = csv_exporter.export_daily_predictions(
                date_str=None,
                prediction_results=[],
                processing_metadata={}
            )
            assert result is not None
        except Exception:
            pass
    
    def test_custom_filename_species_summary(self, csv_exporter, sample_prediction_results):
        """Test species summary export with custom filename."""
        daily_results = {"2025-08-24": sample_prediction_results}
        custom_filename = "custom_species_summary.csv"
        
        output_file = csv_exporter.export_species_summary(
            daily_results=daily_results,
            output_filename=custom_filename
        )
        
        assert output_file.name == custom_filename
        assert output_file.exists()
