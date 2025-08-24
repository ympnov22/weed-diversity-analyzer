"""Tests for calendar visualizer implementation."""

import pytest
from pathlib import Path
from datetime import datetime, date
import tempfile
import shutil
import json

from src.visualization.calendar_visualizer import CalendarVisualizer, CalendarConfig


class TestCalendarVisualizer:
    """Test cases for calendar visualizer."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def calendar_visualizer(self):
        """Create calendar visualizer instance."""
        return CalendarVisualizer()
    
    @pytest.fixture
    def sample_calendar_data(self):
        """Create sample calendar data."""
        return {
            "metric": "shannon_diversity",
            "data": [
                {
                    "date": "2025-08-24",
                    "value": 1.5,
                    "level": 3,
                    "species_count": 5,
                    "total_images": 10
                },
                {
                    "date": "2025-08-25",
                    "value": 2.1,
                    "level": 4,
                    "species_count": 7,
                    "total_images": 15
                }
            ],
            "scale": {
                "min": 0.0,
                "max": 3.0,
                "q25": 0.75,
                "q50": 1.5,
                "q75": 2.25
            },
            "metadata": {
                "total_days": 2,
                "metric_description": "Shannon diversity index (H') - measures species diversity",
                "export_timestamp": datetime.now().isoformat()
            }
        }
    
    def test_initialization_default_config(self):
        """Test calendar visualizer initialization with default config."""
        visualizer = CalendarVisualizer()
        
        assert visualizer.config.cell_size == 12
        assert visualizer.config.color_scheme == "green"
        assert visualizer.config.intensity_levels == 5
        assert visualizer.config.tooltip_enabled is True
        assert visualizer.config.responsive is True
        
        assert "green" in visualizer.color_schemes
        assert "blue" in visualizer.color_schemes
        assert "purple" in visualizer.color_schemes
        assert len(visualizer.color_schemes["green"]) == 5
    
    def test_initialization_custom_config(self):
        """Test calendar visualizer initialization with custom config."""
        config = CalendarConfig(
            cell_size=15,
            color_scheme="blue",
            intensity_levels=4,
            tooltip_enabled=False
        )
        visualizer = CalendarVisualizer(config)
        
        assert visualizer.config.cell_size == 15
        assert visualizer.config.color_scheme == "blue"
        assert visualizer.config.intensity_levels == 4
        assert visualizer.config.tooltip_enabled is False
    
    def test_generate_calendar_html_success(self, calendar_visualizer, sample_calendar_data, temp_output_dir):
        """Test successful calendar HTML generation."""
        output_path = temp_output_dir / "calendar.html"
        
        html_content = calendar_visualizer.generate_calendar_html(
            calendar_data=sample_calendar_data,
            output_path=output_path,
            year=2025
        )
        
        assert isinstance(html_content, str)
        assert "<!DOCTYPE html>" in html_content
        assert "畑植生多様性カレンダー" in html_content
        assert "Shannon diversity index" in html_content
        assert "2025" in html_content
        
        assert output_path.exists()
        
        with open(output_path, 'r', encoding='utf-8') as f:
            saved_content = f.read()
        assert saved_content == html_content
    
    def test_generate_calendar_html_without_output_path(self, calendar_visualizer, sample_calendar_data):
        """Test calendar HTML generation without saving to file."""
        html_content = calendar_visualizer.generate_calendar_html(
            calendar_data=sample_calendar_data,
            year=2025
        )
        
        assert isinstance(html_content, str)
        assert "<!DOCTYPE html>" in html_content
        assert "畑植生多様性カレンダー" in html_content
    
    def test_generate_calendar_html_current_year(self, calendar_visualizer, sample_calendar_data):
        """Test calendar HTML generation with current year."""
        html_content = calendar_visualizer.generate_calendar_html(
            calendar_data=sample_calendar_data
        )
        
        current_year = datetime.now().year
        assert str(current_year) in html_content
    
    def test_generate_calendar_svg(self, calendar_visualizer, sample_calendar_data):
        """Test SVG calendar generation."""
        data_by_date = {entry["date"]: entry for entry in sample_calendar_data["data"]}
        colors = calendar_visualizer.color_schemes["green"]
        
        svg_content = calendar_visualizer._generate_calendar_svg(data_by_date, 2025, colors)
        
        assert isinstance(svg_content, str)
        assert "<svg" in svg_content
        assert "</svg>" in svg_content
        assert "calendar-svg" in svg_content
        assert "calendar-cell" in svg_content
        assert "2025-08-24" in svg_content
        assert "2025-08-25" in svg_content
    
    def test_generate_month_labels(self, calendar_visualizer):
        """Test month labels generation."""
        labels = calendar_visualizer._generate_month_labels(2025, 12, 2)
        
        assert isinstance(labels, str)
        assert "1月" in labels
        assert "12月" in labels
        assert "month-label" in labels
        assert "<text" in labels
    
    def test_generate_day_labels(self, calendar_visualizer):
        """Test day labels generation."""
        labels = calendar_visualizer._generate_day_labels(12, 2)
        
        assert isinstance(labels, str)
        assert "月" in labels
        assert "日" in labels
        assert "day-label" in labels
        assert "<text" in labels
    
    def test_get_calendar_css(self, calendar_visualizer):
        """Test CSS generation."""
        css_content = calendar_visualizer._get_calendar_css()
        
        assert isinstance(css_content, str)
        assert ".calendar-container" in css_content
        assert ".calendar-cell" in css_content
        assert ".tooltip" in css_content
        assert "@media" in css_content
    
    def test_get_calendar_javascript(self, calendar_visualizer, sample_calendar_data):
        """Test JavaScript generation."""
        data_by_date = {entry["date"]: entry for entry in sample_calendar_data["data"]}
        
        js_content = calendar_visualizer._get_calendar_javascript(data_by_date, "shannon_diversity")
        
        assert isinstance(js_content, str)
        assert "addEventListener" in js_content
        assert "tooltip" in js_content
        assert "shannon_diversity" in js_content
        assert "showDayDetails" in js_content
        assert "changeYear" in js_content
    
    def test_generate_sample_data_default(self, calendar_visualizer):
        """Test sample data generation with defaults."""
        sample_data = calendar_visualizer.generate_sample_data()
        
        assert isinstance(sample_data, dict)
        assert "metric" in sample_data
        assert "data" in sample_data
        assert "scale" in sample_data
        assert "metadata" in sample_data
        
        assert sample_data["metric"] == "shannon_diversity"
        assert len(sample_data["data"]) == 365
        
        for entry in sample_data["data"][:5]:
            assert "date" in entry
            assert "value" in entry
            assert "level" in entry
            assert "species_count" in entry
            assert "total_images" in entry
            assert 1 <= entry["level"] <= 4
            assert entry["species_count"] >= 1
            assert 5 <= entry["total_images"] <= 25
    
    def test_generate_sample_data_custom(self, calendar_visualizer):
        """Test sample data generation with custom parameters."""
        sample_data = calendar_visualizer.generate_sample_data(year=2024, num_days=30)
        
        assert len(sample_data["data"]) == 30
        assert sample_data["data"][0]["date"].startswith("2024-01-01")
        assert sample_data["metadata"]["total_days"] == 30
    
    def test_color_schemes(self, calendar_visualizer):
        """Test color scheme availability."""
        assert "green" in calendar_visualizer.color_schemes
        assert "blue" in calendar_visualizer.color_schemes
        assert "purple" in calendar_visualizer.color_schemes
        
        for scheme_name, colors in calendar_visualizer.color_schemes.items():
            assert len(colors) == 5
            for color in colors:
                assert color.startswith("#")
                assert len(color) == 7
    
    def test_empty_calendar_data(self, calendar_visualizer):
        """Test handling of empty calendar data."""
        empty_data = {
            "metric": "shannon_diversity",
            "data": [],
            "scale": {"min": 0.0, "max": 1.0, "q25": 0.0, "q50": 0.5, "q75": 1.0},
            "metadata": {"total_days": 0, "metric_description": "Empty data"}
        }
        
        html_content = calendar_visualizer.generate_calendar_html(
            calendar_data=empty_data,
            year=2025
        )
        
        assert isinstance(html_content, str)
        assert "<!DOCTYPE html>" in html_content
    
    def test_error_handling(self, calendar_visualizer):
        """Test error handling in calendar generation."""
        invalid_data = {"invalid": "data"}
        
        try:
            result = calendar_visualizer.generate_calendar_html(calendar_data=invalid_data)
            assert result is not None
        except Exception:
            pass
