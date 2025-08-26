"""Database connection and session management."""

import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import Session
from typing import Generator

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./weed_diversity.db")

if "sqlite" in DATABASE_URL:
    engine = create_engine(
        DATABASE_URL,
        connect_args={
            "check_same_thread": False,
            "timeout": 20
        },
        poolclass=StaticPool,
        pool_pre_ping=True
    )
else:
    engine = create_engine(
        DATABASE_URL,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_pre_ping=True
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base: DeclarativeMeta = declarative_base()

def get_db() -> Generator[Session, None, None]:
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db() -> None:
    """Initialize database tables."""
    from . import models
    Base.metadata.create_all(bind=engine)
