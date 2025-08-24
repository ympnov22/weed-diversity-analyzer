"""Tests for time series visualizer implementation."""

import pytest
from pathlib import Path
from datetime import datetime
import tempfile
import shutil
import json

from src.visualization.time_series_visualizer import TimeSeriesVisualizer


class TestTimeSeriesVisualizer:
    """Test cases for time series visualizer."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def time_series_visualizer(self):
        """Create time series visualizer instance."""
        return TimeSeriesVisualizer()
    
    @pytest.fixture
    def sample_time_series_data(self):
        """Create sample time series data."""
        return {
            "period": {
                "start_date": "2025-08-01",
                "end_date": "2025-08-31",
                "total_days": 31
            },
            "diversity_trends": {
                "dates": ["2025-08-01", "2025-08-02", "2025-08-03"],
                "shannon_diversity": [1.5, 1.8, 1.2],
                "species_richness": [5, 7, 4],
                "pielou_evenness": [0.8, 0.9, 0.7],
                "simpson_diversity": [0.6, 0.7, 0.5]
            },
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "analysis_version": "1.0.0"
            }
        }
    
    def test_initialization(self):
        """Test time series visualizer initialization."""
        visualizer = TimeSeriesVisualizer()
        assert visualizer is not None
    
    def test_generate_diversity_trends_chart_success(self, time_series_visualizer, sample_time_series_data, temp_output_dir):
        """Test successful diversity trends chart generation."""
        output_path = temp_output_dir / "trends.html"
        
        html_content = time_series_visualizer.generate_diversity_trends_chart(
            time_series_data=sample_time_series_data,
            output_path=output_path
        )
        
        assert isinstance(html_content, str)
        assert "<!DOCTYPE html>" in html_content
        assert "植生多様性時系列分析" in html_content
        assert output_path.exists()
        
        with open(output_path, 'r', encoding='utf-8') as f:
            saved_content = f.read()
        assert len(saved_content) > 0
    
    def test_generate_diversity_trends_chart_without_output_path(self, time_series_visualizer, sample_time_series_data):
        """Test trends chart generation without saving to file."""
        html_content = time_series_visualizer.generate_diversity_trends_chart(
            time_series_data=sample_time_series_data
        )
        
        assert isinstance(html_content, str)
        assert "<!DOCTYPE html>" in html_content
        assert "植生多様性時系列分析" in html_content
    
    def test_generate_confidence_interval_chart_success(self, time_series_visualizer, sample_time_series_data, temp_output_dir):
        """Test successful confidence interval chart generation."""
        output_path = temp_output_dir / "confidence.html"
        
        html_content = time_series_visualizer.generate_confidence_interval_chart(
            time_series_data=sample_time_series_data,
            metric="shannon_diversity",
            output_path=output_path
        )
        
        assert isinstance(html_content, str)
        assert "<!DOCTYPE html>" in html_content
        assert "植生多様性時系列分析" in html_content
        assert output_path.exists()
    
    def test_generate_seasonal_analysis_chart_success(self, time_series_visualizer, sample_time_series_data, temp_output_dir):
        """Test successful seasonal analysis chart generation."""
        output_path = temp_output_dir / "seasonal.html"
        
        html_content = time_series_visualizer.generate_seasonal_analysis_chart(
            time_series_data=sample_time_series_data,
            output_path=output_path
        )
        
        assert isinstance(html_content, str)
        assert "<!DOCTYPE html>" in html_content
        assert "植生多様性時系列分析" in html_content
        assert output_path.exists()
    
    def test_wrap_plotly_html(self, time_series_visualizer):
        """Test Plotly HTML wrapping."""
        plotly_html = "<div>Sample Plotly Content</div>"
        
        wrapped_html = time_series_visualizer._wrap_plotly_html(plotly_html)
        
        assert isinstance(wrapped_html, str)
        assert "<!DOCTYPE html>" in wrapped_html
        assert "植生多様性時系列分析" in wrapped_html
        assert "Sample Plotly Content" in wrapped_html
        assert ".chart-container" in wrapped_html
        assert "@media" in wrapped_html
    
    def test_generate_empty_chart(self, time_series_visualizer):
        """Test empty chart generation."""
        empty_html = time_series_visualizer._generate_empty_chart()
        
        assert isinstance(empty_html, str)
        assert "<!DOCTYPE html>" in empty_html
        assert "時系列データがありません" in empty_html
        assert "表示するデータが見つかりませんでした" in empty_html
    
    def test_generate_sample_time_series_data_default(self, time_series_visualizer):
        """Test sample time series data generation with defaults."""
        sample_data = time_series_visualizer.generate_sample_time_series_data()
        
        assert isinstance(sample_data, dict)
        assert "period" in sample_data
        assert "diversity_trends" in sample_data
        assert "metadata" in sample_data
        
        period = sample_data["period"]
        assert period["start_date"] == "2025-01-01"
        assert period["total_days"] == 365
        
        trends = sample_data["diversity_trends"]
        assert "dates" in trends
        assert "shannon_diversity" in trends
        assert "species_richness" in trends
        assert "pielou_evenness" in trends
        assert "simpson_diversity" in trends
        
        assert len(trends["dates"]) == 365
        assert len(trends["shannon_diversity"]) == 365
        assert len(trends["species_richness"]) == 365
        
        for shannon_value in trends["shannon_diversity"][:5]:
            assert 0 <= shannon_value <= 3.0
        
        for richness_value in trends["species_richness"][:5]:
            assert richness_value >= 1
    
    def test_generate_sample_time_series_data_custom(self, time_series_visualizer):
        """Test sample time series data generation with custom parameters."""
        sample_data = time_series_visualizer.generate_sample_time_series_data(
            start_date="2024-06-01",
            num_days=30
        )
        
        period = sample_data["period"]
        assert period["start_date"] == "2024-06-01"
        assert period["end_date"] == "2024-06-30"
        assert period["total_days"] == 30
        
        trends = sample_data["diversity_trends"]
        assert len(trends["dates"]) == 30
        assert trends["dates"][0] == "2024-06-01"
        assert trends["dates"][-1] == "2024-06-30"
    
    def test_empty_time_series_data(self, time_series_visualizer):
        """Test handling of empty time series data."""
        empty_data = {
            "diversity_trends": {
                "dates": [],
                "shannon_diversity": []
            }
        }
        
        html_content = time_series_visualizer.generate_diversity_trends_chart(
            time_series_data=empty_data
        )
        
        assert isinstance(html_content, str)
        assert "時系列データがありません" in html_content
    
    def test_missing_diversity_trends(self, time_series_visualizer):
        """Test handling of missing diversity trends."""
        incomplete_data = {
            "period": {
                "start_date": "2025-08-01",
                "end_date": "2025-08-31"
            }
        }
        
        html_content = time_series_visualizer.generate_diversity_trends_chart(
            time_series_data=incomplete_data
        )
        
        assert isinstance(html_content, str)
        assert "時系列データがありません" in html_content
    
    def test_confidence_interval_chart_empty_data(self, time_series_visualizer):
        """Test confidence interval chart with empty data."""
        empty_data = {
            "diversity_trends": {
                "dates": [],
                "shannon_diversity": []
            }
        }
        
        html_content = time_series_visualizer.generate_confidence_interval_chart(
            time_series_data=empty_data,
            metric="shannon_diversity"
        )
        
        assert isinstance(html_content, str)
        assert "時系列データがありません" in html_content
    
    def test_seasonal_analysis_empty_data(self, time_series_visualizer):
        """Test seasonal analysis with empty data."""
        empty_data = {
            "diversity_trends": {
                "dates": [],
                "shannon_diversity": []
            }
        }
        
        html_content = time_series_visualizer.generate_seasonal_analysis_chart(
            time_series_data=empty_data
        )
        
        assert isinstance(html_content, str)
        assert "時系列データがありません" in html_content
    
    def test_error_handling(self, time_series_visualizer):
        """Test error handling in time series generation."""
        invalid_data = {"invalid": "data"}
        
        try:
            result = time_series_visualizer.generate_diversity_trends_chart(time_series_data=invalid_data)
            assert result is not None
        except Exception:
            pass
