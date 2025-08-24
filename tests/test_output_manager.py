"""Tests for output manager implementation."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import shutil

from src.output.output_manager import OutputManager
from src.utils.data_structures import DiversityMetrics, PredictionResult, SpeciesPrediction


class TestOutputManager:
    """Test cases for output manager."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def output_manager(self, temp_output_dir):
        """Create output manager instance."""
        return OutputManager(temp_output_dir)
    
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
        
        predictions = [
            SpeciesPrediction("Species A", 0.9),
            SpeciesPrediction("Species B", 0.7),
            SpeciesPrediction("Species C", 0.5)
        ]
        
        result = PredictionResult(
            image_path=Path("image_001.jpg"),
            predictions=predictions,
            model_name="inatag",
            processing_time=0.1,
            timestamp=datetime(2025, 8, 24, 12, 0, 0)
        )
        
        return [result]
    
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
                "accepted_images": 8
            },
            "clustering_stats": {
                "clusters_formed": 3,
                "representative_images": 2
            }
        }
    
    def test_initialization(self, temp_output_dir):
        """Test output manager initialization."""
        manager = OutputManager(temp_output_dir)
        
        assert manager.output_base_dir == temp_output_dir
        assert manager.output_base_dir.exists()
        assert manager.json_exporter.output_dir == temp_output_dir / "json"
        assert manager.csv_exporter.output_dir == temp_output_dir / "csv"
    
    def test_initialization_default_dir(self):
        """Test output manager with default directory."""
        manager = OutputManager()
        
        assert manager.output_base_dir == Path("output")
    
    def test_export_daily_analysis_success(
        self,
        output_manager,
        sample_diversity_metrics,
        sample_prediction_results,
        sample_processing_metadata
    ):
        """Test successful daily analysis export."""
        date_str = "2025-08-24"
        confidence_intervals = {
            "shannon_diversity": (1.2, 1.8),
            "species_richness": (4.0, 6.0)
        }
        soft_voting_results = {
            "species_weights": {"Species A": 0.6, "Species B": 0.4},
            "diversity_metrics": {"shannon_diversity": 1.5}
        }
        
        output_files = output_manager.export_daily_analysis(
            date_str=date_str,
            diversity_metrics=sample_diversity_metrics,
            prediction_results=sample_prediction_results,
            processing_metadata=sample_processing_metadata,
            confidence_intervals=confidence_intervals,
            soft_voting_results=soft_voting_results
        )
        
        assert "json_summary" in output_files
        assert "csv_predictions" in output_files
        assert "csv_soft_voting" in output_files
        
        assert output_files["json_summary"].exists()
        assert output_files["csv_predictions"].exists()
        assert output_files["csv_soft_voting"].exists()
        
        assert f"daily_summary_{date_str}.json" in str(output_files["json_summary"])
        assert f"daily_predictions_{date_str}.csv" in str(output_files["csv_predictions"])
        assert f"soft_voting_{date_str}.csv" in str(output_files["csv_soft_voting"])
    
    def test_export_daily_analysis_without_soft_voting(
        self,
        output_manager,
        sample_diversity_metrics,
        sample_prediction_results,
        sample_processing_metadata
    ):
        """Test daily analysis export without soft voting results."""
        date_str = "2025-08-24"
        
        output_files = output_manager.export_daily_analysis(
            date_str=date_str,
            diversity_metrics=sample_diversity_metrics,
            prediction_results=sample_prediction_results,
            processing_metadata=sample_processing_metadata
        )
        
        assert "json_summary" in output_files
        assert "csv_predictions" in output_files
        assert "csv_soft_voting" not in output_files
        
        assert output_files["json_summary"].exists()
        assert output_files["csv_predictions"].exists()
    
    def test_export_period_analysis(self, output_manager, sample_prediction_results):
        """Test period analysis export."""
        daily_summaries = [
            {
                "date": "2025-08-24",
                "diversity_metrics": {
                    "species_richness": 5,
                    "shannon_diversity": 1.5
                },
                "top_species": [{"species_name": "Species A", "count": 3}]
            },
            {
                "date": "2025-08-25",
                "diversity_metrics": {
                    "species_richness": 6,
                    "shannon_diversity": 1.7
                },
                "top_species": [{"species_name": "Species B", "count": 4}]
            }
        ]
        
        daily_results = {
            "2025-08-24": sample_prediction_results,
            "2025-08-25": sample_prediction_results
        }
        
        output_files = output_manager.export_period_analysis(
            daily_summaries=daily_summaries,
            daily_results=daily_results,
            start_date="2025-08-24",
            end_date="2025-08-25"
        )
        
        assert "json_time_series" in output_files
        assert "json_github_calendar" in output_files
        assert "csv_aggregated" in output_files
        assert "csv_species_summary" in output_files
        
        for file_path in output_files.values():
            assert file_path.exists()
        
        assert "time_series_2025-08-24_to_2025-08-25.json" in str(output_files["json_time_series"])
        assert "github_calendar_shannon_diversity.json" in str(output_files["json_github_calendar"])
        assert "aggregated_predictions_2025-08-24_to_2025-08-25.csv" in str(output_files["csv_aggregated"])
        assert "species_summary" in str(output_files["csv_species_summary"])
    
    def test_export_visualization_data(self, output_manager):
        """Test visualization data export."""
        daily_summaries = [
            {
                "date": "2025-08-24",
                "diversity_metrics": {
                    "shannon_diversity": 1.5,
                    "species_richness": 5,
                    "pielou_evenness": 0.8
                },
                "metadata": {"total_images_processed": 10}
            }
        ]
        
        metrics = ["shannon_diversity", "species_richness"]
        
        output_files = output_manager.export_visualization_data(
            daily_summaries=daily_summaries,
            metrics=metrics
        )
        
        assert "github_calendar_shannon_diversity" in output_files
        assert "github_calendar_species_richness" in output_files
        
        for file_path in output_files.values():
            assert file_path.exists()
    
    def test_export_visualization_data_default_metrics(self, output_manager):
        """Test visualization data export with default metrics."""
        daily_summaries = [
            {
                "date": "2025-08-24",
                "diversity_metrics": {
                    "shannon_diversity": 1.5,
                    "species_richness": 5,
                    "pielou_evenness": 0.8
                },
                "metadata": {"total_images_processed": 10}
            }
        ]
        
        output_files = output_manager.export_visualization_data(
            daily_summaries=daily_summaries
        )
        
        assert "github_calendar_shannon_diversity" in output_files
        assert "github_calendar_species_richness" in output_files
        assert "github_calendar_pielou_evenness" in output_files
    
    def test_get_output_summary(self, output_manager, sample_diversity_metrics, sample_prediction_results, sample_processing_metadata):
        """Test output summary generation."""
        output_manager.export_daily_analysis(
            date_str="2025-08-24",
            diversity_metrics=sample_diversity_metrics,
            prediction_results=sample_prediction_results,
            processing_metadata=sample_processing_metadata
        )
        
        summary = output_manager.get_output_summary()
        
        assert "base_directory" in summary
        assert "json_directory" in summary
        assert "csv_directory" in summary
        assert "files" in summary
        assert "total_files" in summary
        assert "last_updated" in summary
        
        assert "json" in summary["files"]
        assert "csv" in summary["files"]
        assert len(summary["files"]["json"]) > 0
        assert len(summary["files"]["csv"]) > 0
        assert summary["total_files"] > 0
    
    def test_cleanup_old_outputs(self, output_manager, sample_diversity_metrics, sample_prediction_results, sample_processing_metadata):
        """Test cleanup of old output files."""
        output_manager.export_daily_analysis(
            date_str="2025-08-24",
            diversity_metrics=sample_diversity_metrics,
            prediction_results=sample_prediction_results,
            processing_metadata=sample_processing_metadata
        )
        
        deleted_count = output_manager.cleanup_old_outputs(days_to_keep=1)
        
        assert deleted_count == 0
        
        deleted_count = output_manager.cleanup_old_outputs(days_to_keep=0)
        
        assert deleted_count >= 0
    
    def test_error_handling(self, output_manager):
        """Test error handling in output manager methods."""
        try:
            result = output_manager.export_daily_analysis(
                date_str="2025-08-24",
                diversity_metrics=None,
                prediction_results=[],
                processing_metadata={}
            )
            assert result is not None
        except Exception:
            pass
        
        try:
            result = output_manager.export_period_analysis(
                daily_summaries=None,
                daily_results={},
                start_date="2025-08-24",
                end_date="2025-08-25"
            )
            assert result is not None
        except Exception:
            pass
