"""Tests for web server authentication integration."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch
import os

from src.visualization.web_server import WebServer
from src.output.output_manager import OutputManager
from src.database.database import Base, get_db
from src.database.models import UserModel, AnalysisSessionModel
from src.database.services import DatabaseService

@pytest.fixture
def test_db():
    """Create test database."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    def override_get_db():
        try:
            db = TestingSessionLocal()
            yield db
        finally:
            db.close()
    
    return override_get_db

@pytest.fixture
def test_user(test_db):
    """Create test user."""
    db = next(test_db())
    user = UserModel(
        username="testuser",
        email="test@example.com",
        api_key="test-api-key-123",
        is_active=True
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@pytest.fixture
def client(test_db):
    """Create test client with database override."""
    output_manager = OutputManager()
    web_server = WebServer(output_manager)
    app = web_server.get_app()
    
    app.dependency_overrides[get_db] = test_db
    
    with patch.dict(os.environ, {"DATABASE_URL": "sqlite:///:memory:"}):
        return TestClient(app)

class TestWebServerAuth:
    """Test web server authentication integration."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_calendar_api_without_auth(self, client):
        """Test calendar API without authentication returns sample data."""
        response = client.get("/api/calendar/shannon_diversity")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "metric" in data
    
    def test_calendar_api_with_invalid_auth(self, client):
        """Test calendar API with invalid authentication."""
        headers = {"Authorization": "Bearer invalid-key"}
        response = client.get("/api/calendar/shannon_diversity", headers=headers)
        assert response.status_code == 200
    
    def test_calendar_api_with_valid_auth(self, client, test_user):
        """Test calendar API with valid authentication."""
        headers = {"Authorization": f"Bearer {test_user.api_key}"}
        response = client.get("/api/calendar/shannon_diversity", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
    
    def test_sessions_api_requires_auth(self, client):
        """Test sessions API requires authentication."""
        response = client.get("/api/sessions")
        assert response.status_code == 401
    
    def test_sessions_api_with_auth(self, client, test_user):
        """Test sessions API with authentication."""
        headers = {"Authorization": f"Bearer {test_user.api_key}"}
        response = client.get("/api/sessions", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_create_session_requires_auth(self, client):
        """Test create session requires authentication."""
        response = client.post("/api/sessions", json={"session_name": "test"})
        assert response.status_code == 401
    
    def test_create_session_with_auth(self, client, test_user):
        """Test create session with authentication."""
        headers = {"Authorization": f"Bearer {test_user.api_key}"}
        response = client.post("/api/sessions", 
                             headers=headers,
                             json={"session_name": "test session", "description": "test"})
        assert response.status_code == 201
        data = response.json()
        assert data["session_name"] == "test session"
        assert "id" in data
    
    def test_x_api_key_header(self, client, test_user):
        """Test X-API-Key header authentication."""
        headers = {"X-API-Key": test_user.api_key}
        response = client.get("/api/sessions", headers=headers)
        assert response.status_code == 200
