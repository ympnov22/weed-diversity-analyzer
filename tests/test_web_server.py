"""Tests for web server implementation."""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import shutil
import json

from src.visualization.web_server import WebServer
from src.output.output_manager import OutputManager


class TestWebServer:
    """Test cases for web server."""
    
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
    def web_server(self, output_manager):
        """Create web server instance."""
        return WebServer(output_manager=output_manager)
    
    @pytest.fixture
    def test_client(self, web_server):
        """Create test client."""
        return TestClient(web_server.get_app())
    
    def test_initialization(self, output_manager):
        """Test web server initialization."""
        server = WebServer(output_manager=output_manager, host="localhost", port=8080)
        
        assert server.output_manager == output_manager
        assert server.host == "localhost"
        assert server.port == 8080
        assert server.calendar_viz is not None
        assert server.time_series_viz is not None
        assert server.dashboard_gen is not None
        assert server.static_dir.exists()
    
    def test_initialization_default_params(self):
        """Test web server initialization with default parameters."""
        server = WebServer()
        
        assert server.host == "127.0.0.1"
        assert server.port == 8000
        assert server.output_manager is not None
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        
        content = response.text
        assert "<!DOCTYPE html>" in content
        assert "畑植生多様性解析ダッシュボード" in content
        assert "iNatAg (2,959種対応)" in content
        assert "カレンダー表示" in content
        assert "時系列分析" in content
        assert "種分析" in content
        assert "モデル性能" in content
    
    def test_calendar_data_endpoint(self, test_client):
        """Test calendar data API endpoint."""
        response = test_client.get("/api/calendar/shannon_diversity")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        data = response.json()
        assert "metric" in data
        assert "data" in data
        assert "scale" in data
        assert "metadata" in data
        assert data["metric"] == "shannon_diversity"
        assert isinstance(data["data"], list)
    
    def test_time_series_data_endpoint(self, test_client):
        """Test time series data API endpoint."""
        response = test_client.get("/api/time-series")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        data = response.json()
        assert "period" in data
        assert "diversity_trends" in data
        assert "metadata" in data
        assert isinstance(data["diversity_trends"], dict)
    
    def test_time_series_data_with_params(self, test_client):
        """Test time series data endpoint with date parameters."""
        response = test_client.get("/api/time-series?start_date=2025-08-01&end_date=2025-08-31")
        
        assert response.status_code == 200
        data = response.json()
        assert "period" in data
        assert "diversity_trends" in data
    
    def test_species_dashboard_data_endpoint(self, test_client):
        """Test species dashboard data API endpoint."""
        response = test_client.get("/api/dashboard/species")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        data = response.json()
        assert isinstance(data, list)
        if data:
            assert "date" in data[0]
            assert "diversity_metrics" in data[0]
            assert "top_species" in data[0]
    
    def test_status_endpoint(self, test_client):
        """Test status API endpoint."""
        response = test_client.get("/api/status")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        data = response.json()
        assert "server" in data
        assert "timestamp" in data
        assert "version" in data
        assert "inatag_species_count" in data
        assert "output_summary" in data
        
        assert data["server"] == "running"
        assert data["version"] == "1.0.0"
        assert data["inatag_species_count"] == 2959
    
    def test_calendar_page_endpoint(self, test_client):
        """Test calendar page endpoint."""
        response = test_client.get("/calendar")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        
        content = response.text
        assert "<!DOCTYPE html>" in content
        assert "畑植生多様性カレンダー" in content
        assert "calendar-container" in content
    
    def test_calendar_page_with_params(self, test_client):
        """Test calendar page with parameters."""
        response = test_client.get("/calendar?metric=species_richness&year=2024")
        
        assert response.status_code == 200
        content = response.text
        assert "<!DOCTYPE html>" in content
        assert "畑植生多様性カレンダー" in content
    
    def test_time_series_page_endpoint(self, test_client):
        """Test time series page endpoint."""
        response = test_client.get("/time-series")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        
        content = response.text
        assert "<!DOCTYPE html>" in content
        assert "植生多様性時系列分析" in content
    
    def test_species_dashboard_page_endpoint(self, test_client):
        """Test species dashboard page endpoint."""
        response = test_client.get("/dashboard/species")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        
        content = response.text
        assert "<!DOCTYPE html>" in content
        assert "iNatAg植生種分析ダッシュボード" in content
    
    def test_performance_dashboard_page_endpoint(self, test_client):
        """Test performance dashboard page endpoint."""
        response = test_client.get("/dashboard/performance")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        
        content = response.text
        assert "<!DOCTYPE html>" in content
        assert "Swin Transformer性能分析" in content
    
    def test_invalid_calendar_metric(self, test_client):
        """Test invalid calendar metric handling."""
        response = test_client.get("/api/calendar/invalid_metric")
        
        assert response.status_code == 200
        data = response.json()
        assert "metric" in data
        assert "data" in data
    
    def test_get_app_method(self, web_server):
        """Test get_app method."""
        app = web_server.get_app()
        
        assert app is not None
        assert hasattr(app, "routes")
        assert len(app.routes) > 0
    
    def test_static_directory_creation(self, output_manager):
        """Test static directory creation."""
        server = WebServer(output_manager=output_manager)
        
        assert server.static_dir.exists()
        assert server.static_dir.is_dir()
    
    def test_dashboard_css_generation(self, web_server):
        """Test dashboard CSS generation."""
        css_content = web_server._get_dashboard_css()
        
        assert isinstance(css_content, str)
        assert ".dashboard-container" in css_content
        assert ".nav-card" in css_content
        assert ".info-card" in css_content
        assert "@media" in css_content
    
    def test_dashboard_javascript_generation(self, web_server):
        """Test dashboard JavaScript generation."""
        js_content = web_server._get_dashboard_javascript()
        
        assert isinstance(js_content, str)
        assert "addEventListener" in js_content
        assert "fetch" in js_content
        assert "/api/status" in js_content
    
    def test_main_dashboard_html_generation(self, web_server):
        """Test main dashboard HTML generation."""
        html_content = web_server._get_main_dashboard_html()
        
        assert isinstance(html_content, str)
        assert "<!DOCTYPE html>" in html_content
        assert "畑植生多様性解析ダッシュボード" in html_content
        assert "iNatAg (2,959種対応)" in html_content
        assert "/calendar" in html_content
        assert "/time-series" in html_content
        assert "/dashboard/species" in html_content
        assert "/dashboard/performance" in html_content
    
    def test_nonexistent_endpoint(self, test_client):
        """Test nonexistent endpoint handling."""
        response = test_client.get("/nonexistent")
        
        assert response.status_code == 404
