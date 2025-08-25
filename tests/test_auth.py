"""Tests for authentication system."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.database import Base
from src.database.models import UserModel
from src.auth.auth import create_api_key, verify_api_key, get_current_user
from fastapi import HTTPException


@pytest.fixture
def db_session():
    """Create test database session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


def test_create_api_key():
    """Test API key creation."""
    api_key = create_api_key()
    assert isinstance(api_key, str)
    assert len(api_key) > 20


def test_verify_api_key_valid(db_session):
    """Test API key verification with valid key."""
    api_key = create_api_key()
    user = UserModel(
        username="testuser",
        email="test@example.com",
        api_key=api_key,
        is_active=True
    )
    db_session.add(user)
    db_session.commit()
    
    verified_user = verify_api_key(api_key, db_session)
    assert verified_user is not None
    assert verified_user.username == "testuser"


def test_verify_api_key_invalid(db_session):
    """Test API key verification with invalid key."""
    verified_user = verify_api_key("invalid_key", db_session)
    assert verified_user is None


def test_verify_api_key_inactive_user(db_session):
    """Test API key verification with inactive user."""
    api_key = create_api_key()
    user = UserModel(
        username="testuser",
        email="test@example.com",
        api_key=api_key,
        is_active=False
    )
    db_session.add(user)
    db_session.commit()
    
    verified_user = verify_api_key(api_key, db_session)
    assert verified_user is None


def test_get_current_user_valid(db_session):
    """Test getting current user with valid API key."""
    api_key = create_api_key()
    user = UserModel(
        username="testuser",
        email="test@example.com",
        api_key=api_key,
        is_active=True
    )
    db_session.add(user)
    db_session.commit()
    
    current_user = get_current_user(api_key, db_session)
    assert current_user is not None
    assert current_user.username == "testuser"


def test_get_current_user_invalid(db_session):
    """Test getting current user with invalid API key."""
    with pytest.raises(HTTPException) as exc_info:
        get_current_user("invalid_key", db_session)
    
    assert exc_info.value.status_code == 401
    assert "Invalid API key" in str(exc_info.value.detail)
