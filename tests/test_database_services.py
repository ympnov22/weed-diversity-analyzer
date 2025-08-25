"""Tests for database services."""

import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.database import Base
from src.database.models import UserModel, AnalysisSessionModel
from src.database.services import DatabaseService
from src.utils.data_structures import DiversityMetrics


@pytest.fixture
def db_session():
    """Create test database session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def sample_user(db_session):
    """Create sample user for testing."""
    user = UserModel(
        username="testuser",
        email="test@example.com",
        api_key="test_api_key_123"
    )
    db_session.add(user)
    db_session.commit()
    return user


@pytest.fixture
def sample_session(db_session, sample_user):
    """Create sample analysis session."""
    session = AnalysisSessionModel(
        user_id=sample_user.id,
        session_name="Test Session",
        start_date=datetime.now()
    )
    db_session.add(session)
    db_session.commit()
    return session


def test_database_service_initialization(db_session):
    """Test DatabaseService initialization."""
    service = DatabaseService(db_session)
    assert service.db == db_session


def test_save_daily_analysis(db_session, sample_user, sample_session):
    """Test saving daily analysis to database."""
    service = DatabaseService(db_session)
    
    diversity_metrics = DiversityMetrics(
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
    
    result = service.save_daily_analysis(
        user_id=sample_user.id,
        session_id=sample_session.id,
        date_str="2025-01-01",
        diversity_metrics=diversity_metrics,
        prediction_results=[]
    )
    
    assert result is not None
    assert result.species_richness == 5
    assert result.shannon_diversity == 1.5


def test_get_daily_summaries(db_session, sample_user, sample_session):
    """Test retrieving daily summaries."""
    service = DatabaseService(db_session)
    
    diversity_metrics = DiversityMetrics(
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
    
    service.save_daily_analysis(
        user_id=sample_user.id,
        session_id=sample_session.id,
        date_str="2025-01-01",
        diversity_metrics=diversity_metrics,
        prediction_results=[]
    )
    
    summaries = service.get_daily_summaries(sample_user.id)
    assert len(summaries) == 1
    assert summaries[0]["species_richness"] == 5
    assert summaries[0]["shannon_diversity"] == 1.5


def test_get_calendar_data(db_session, sample_user, sample_session):
    """Test getting calendar data."""
    service = DatabaseService(db_session)
    
    diversity_metrics = DiversityMetrics(
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
    
    service.save_daily_analysis(
        user_id=sample_user.id,
        session_id=sample_session.id,
        date_str="2025-01-01",
        diversity_metrics=diversity_metrics,
        prediction_results=[]
    )
    
    calendar_data = service.get_calendar_data(sample_user.id, "shannon_diversity")
    assert calendar_data["metric"] == "shannon_diversity"
    assert calendar_data["total_days"] == 1
    assert len(calendar_data["data"]) == 1


def test_create_analysis_session(db_session, sample_user):
    """Test creating analysis session."""
    service = DatabaseService(db_session)
    
    session = service.create_analysis_session(
        user_id=sample_user.id,
        session_name="New Test Session",
        description="Test description"
    )
    
    assert session is not None
    assert session.session_name == "New Test Session"
    assert session.user_id == sample_user.id
