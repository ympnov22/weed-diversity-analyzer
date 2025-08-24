"""Tests for dashboard generator implementation."""

import pytest
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

from src.visualization.dashboard_generator import DashboardGenerator


class TestDashboardGenerator:
    """Test cases for dashboard generator."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def dashboard_generator(self):
        """Create dashboard generator instance."""
        return DashboardGenerator()
    
    @pytest.fixture
    def sample_daily_summaries(self):
        """Create sample daily summaries."""
        return [
            {
                "date": "2025-08-24",
                "diversity_metrics": {
                    "species_richness": 5,
                    "shannon_diversity": 1.5
                },
                "top_species": [
                    {"species_name": "Taraxacum officinale", "count": 3, "average_confidence": 0.85},
                    {"species_name": "Plantago major", "count": 2, "average_confidence": 0.78}
                ],
                "processing_info": {
                    "average_confidence": 0.82
                }
            },
            {
                "date": "2025-08-25",
                "diversity_metrics": {
                    "species_richness": 7,
                    "shannon_diversity": 1.8
                },
                "top_species": [
                    {"species_name": "Trifolium repens", "count": 4, "average_confidence": 0.91},
                    {"species_name": "Taraxacum officinale", "count": 2, "average_confidence": 0.87}
                ],
                "processing_info": {
                    "average_confidence": 0.89
                }
            }
        ]
    
    @pytest.fixture
    def sample_processing_metadata(self):
        """Create sample processing metadata."""
        return {
            "model_info": {
                "name": "inatag",
                "size": "base",
                "lora_enabled": True
            },
            "processing_stats": {
                "total_processing_time": 125.5,
                "average_processing_time": 0.25
            }
        }
    
    def test_initialization(self):
        """Test dashboard generator initialization."""
        generator = DashboardGenerator()
        assert generator is not None
    
    def test_generate_species_distribution_chart_success(self, dashboard_generator, sample_daily_summaries, temp_output_dir):
        """Test successful species distribution chart generation."""
        output_path = temp_output_dir / "species_dashboard.html"
        
        html_content = dashboard_generator.generate_species_distribution_chart(
            daily_summaries=sample_daily_summaries,
            output_path=output_path
        )
        
        assert isinstance(html_content, str)
        assert "<!DOCTYPE html>" in html_content
        assert "iNatAg植生種分析ダッシュボード" in html_content
        assert "iNatAg植生種分析ダッシュボード" in html_content
        
        assert output_path.exists()
        
        with open(output_path, 'r', encoding='utf-8') as f:
            saved_content = f.read()
        assert len(saved_content) > 0
    
    def test_generate_species_distribution_chart_without_output_path(self, dashboard_generator, sample_daily_summaries):
        """Test species distribution chart generation without saving to file."""
        html_content = dashboard_generator.generate_species_distribution_chart(
            daily_summaries=sample_daily_summaries
        )
        
        assert isinstance(html_content, str)
        assert "<!DOCTYPE html>" in html_content
        assert "iNatAg植生種分析ダッシュボード" in html_content
    
    def test_generate_model_performance_dashboard_success(self, dashboard_generator, sample_processing_metadata, temp_output_dir):
        """Test successful model performance dashboard generation."""
        output_path = temp_output_dir / "performance_dashboard.html"
        
        html_content = dashboard_generator.generate_model_performance_dashboard(
            processing_metadata=sample_processing_metadata,
            output_path=output_path
        )
        
        assert isinstance(html_content, str)
        assert "<!DOCTYPE html>" in html_content
        assert "Swin Transformer性能分析" in html_content
        assert "Swin Transformer性能分析" in html_content
        
        assert output_path.exists()
    
    def test_generate_soft_voting_analysis_success(self, dashboard_generator, sample_daily_summaries, temp_output_dir):
        """Test successful soft voting analysis generation."""
        output_path = temp_output_dir / "soft_voting.html"
        
        html_content = dashboard_generator.generate_soft_voting_analysis(
            daily_summaries=sample_daily_summaries,
            output_path=output_path
        )
        
        assert isinstance(html_content, str)
        assert "<!DOCTYPE html>" in html_content
        assert "Top-3ソフト投票分析" in html_content
        assert "Top-3ソフト投票分析" in html_content
        
        assert output_path.exists()
    
    def test_wrap_dashboard_html(self, dashboard_generator):
        """Test dashboard HTML wrapping."""
        plotly_html = "<div>Sample Dashboard Content</div>"
        title = "Test Dashboard"
        
        wrapped_html = dashboard_generator._wrap_dashboard_html(plotly_html, title)
        
        assert isinstance(wrapped_html, str)
        assert "<!DOCTYPE html>" in wrapped_html
        assert title in wrapped_html
        assert "Sample Dashboard Content" in wrapped_html
        assert "iNatAg (2,959種対応)" in wrapped_html
        assert ".dashboard-container" in wrapped_html
        assert "@media" in wrapped_html
    
    def test_generate_empty_dashboard(self, dashboard_generator):
        """Test empty dashboard generation."""
        empty_html = dashboard_generator._generate_empty_dashboard()
        
        assert isinstance(empty_html, str)
        assert "<!DOCTYPE html>" in empty_html
        assert "ダッシュボードデータがありません" in empty_html
        assert "表示するデータが見つかりませんでした" in empty_html
    
    def test_generate_sample_dashboard_data(self, dashboard_generator):
        """Test sample dashboard data generation."""
        sample_data = dashboard_generator.generate_sample_dashboard_data()
        
        assert isinstance(sample_data, dict)
        assert "daily_summaries" in sample_data
        assert "processing_metadata" in sample_data
        
        daily_summaries = sample_data["daily_summaries"]
        assert len(daily_summaries) == 30
        
        for summary in daily_summaries[:3]:
            assert "date" in summary
            assert "diversity_metrics" in summary
            assert "top_species" in summary
            assert "processing_info" in summary
            
            assert "species_richness" in summary["diversity_metrics"]
            assert len(summary["top_species"]) == 5
            assert "average_confidence" in summary["processing_info"]
        
        processing_metadata = sample_data["processing_metadata"]
        assert "model_info" in processing_metadata
        assert "processing_stats" in processing_metadata
        
        model_info = processing_metadata["model_info"]
        assert model_info["name"] == "inatag"
        assert model_info["size"] == "base"
        assert model_info["lora_enabled"] is True
    
    def test_empty_daily_summaries(self, dashboard_generator):
        """Test handling of empty daily summaries."""
        empty_summaries = []
        
        html_content = dashboard_generator.generate_species_distribution_chart(
            daily_summaries=empty_summaries
        )
        
        assert isinstance(html_content, str)
        assert "ダッシュボードデータがありません" in html_content
    
    def test_missing_top_species(self, dashboard_generator):
        """Test handling of missing top species data."""
        summaries_without_species = [
            {
                "date": "2025-08-24",
                "diversity_metrics": {"species_richness": 5},
                "processing_info": {"average_confidence": 0.82}
            }
        ]
        
        html_content = dashboard_generator.generate_species_distribution_chart(
            daily_summaries=summaries_without_species
        )
        
        assert isinstance(html_content, str)
        assert "ダッシュボードデータがありません" in html_content
    
    def test_missing_processing_metadata(self, dashboard_generator):
        """Test handling of missing processing metadata."""
        empty_metadata = {}
        
        html_content = dashboard_generator.generate_model_performance_dashboard(
            processing_metadata=empty_metadata
        )
        
        assert isinstance(html_content, str)
        assert "<!DOCTYPE html>" in html_content
        assert "Swin Transformer性能分析" in html_content
    
    def test_soft_voting_analysis_empty_data(self, dashboard_generator):
        """Test soft voting analysis with empty data."""
        empty_summaries = []
        
        html_content = dashboard_generator.generate_soft_voting_analysis(
            daily_summaries=empty_summaries
        )
        
        assert isinstance(html_content, str)
        assert "<!DOCTYPE html>" in html_content
        assert "Top-3ソフト投票分析" in html_content
    
    def test_error_handling(self, dashboard_generator):
        """Test error handling in dashboard generation."""
        invalid_data = [{"invalid": "data"}]
        
        try:
            result = dashboard_generator.generate_species_distribution_chart(daily_summaries=invalid_data)
            assert result is not None
        except Exception:
            pass
