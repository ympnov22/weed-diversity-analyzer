"""Tests for database models."""

import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.database import Base
from src.database.models import UserModel, AnalysisSessionModel, DiversityMetricsModel


@pytest.fixture
def db_session():
    """Create test database session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


def test_user_model_creation(db_session):
    """Test UserModel creation and basic operations."""
    user = UserModel(
        username="testuser",
        email="test@example.com",
        api_key="test_api_key_123",
        is_active=True
    )
    
    db_session.add(user)
    db_session.commit()
    
    retrieved_user = db_session.query(UserModel).filter_by(username="testuser").first()
    assert retrieved_user is not None
    assert retrieved_user.username == "testuser"
    assert retrieved_user.email == "test@example.com"
    assert retrieved_user.api_key == "test_api_key_123"
    assert retrieved_user.is_active is True


def test_analysis_session_model(db_session):
    """Test AnalysisSessionModel creation and relationships."""
    user = UserModel(
        username="testuser",
        email="test@example.com",
        api_key="test_api_key_123"
    )
    db_session.add(user)
    db_session.commit()
    
    session = AnalysisSessionModel(
        user_id=user.id,
        session_name="Test Session",
        description="Test description",
        start_date=datetime.now()
    )
    
    db_session.add(session)
    db_session.commit()
    
    retrieved_session = db_session.query(AnalysisSessionModel).first()
    assert retrieved_session is not None
    assert retrieved_session.session_name == "Test Session"
    assert retrieved_session.user_id == user.id


def test_diversity_metrics_model(db_session):
    """Test DiversityMetricsModel creation."""
    user = UserModel(
        username="testuser",
        email="test@example.com",
        api_key="test_api_key_123"
    )
    db_session.add(user)
    db_session.commit()
    
    session = AnalysisSessionModel(
        user_id=user.id,
        session_name="Test Session",
        start_date=datetime.now()
    )
    db_session.add(session)
    db_session.commit()
    
    metrics = DiversityMetricsModel(
        session_id=session.id,
        date=datetime.now(),
        total_images=10,
        processed_images=8,
        species_richness=5,
        shannon_diversity=1.5,
        pielou_evenness=0.8,
        simpson_diversity=0.7,
        hill_q0=5.0,
        hill_q1=4.5,
        hill_q2=4.0,
        species_counts={"species1": 3, "species2": 2},
        species_frequencies={"species1": 0.6, "species2": 0.4},
        top_species=[{"name": "species1", "count": 3}]
    )
    
    db_session.add(metrics)
    db_session.commit()
    
    retrieved_metrics = db_session.query(DiversityMetricsModel).first()
    assert retrieved_metrics is not None
    assert retrieved_metrics.species_richness == 5
    assert retrieved_metrics.shannon_diversity == 1.5
    assert retrieved_metrics.species_counts == {"species1": 3, "species2": 2}
