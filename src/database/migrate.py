"""Database migration utilities."""

import logging
import secrets
from sqlalchemy import text
from sqlalchemy.orm import Session
from .database import engine, Base
from .models import UserModel

logger = logging.getLogger(__name__)

def run_migrations() -> None:
    """Run database migrations."""
    try:
        logger.info("Running database migrations...")
        
        Base.metadata.create_all(bind=engine)
        
        _seed_initial_data()
        
        logger.info("Database migrations completed successfully")
        
    except Exception as e:
        logger.error(f"Database migration failed: {e}")
        raise

def _seed_initial_data() -> None:
    """Seed initial data if needed."""
    with Session(engine) as session:
        try:
            existing_user = session.query(UserModel).first()
            if not existing_user:
                api_key = secrets.token_urlsafe(32)
                default_user = UserModel(
                    username="admin",
                    email="admin@example.com",
                    api_key=api_key,
                    is_active=True
                )
                session.add(default_user)
                session.commit()
                
                logger.info(f"Created default user with API key: {api_key}")
                print(f"Default user created with API key: {api_key}")
                
        except Exception as e:
            logger.warning(f"Failed to seed initial data: {e}")
            session.rollback()

if __name__ == "__main__":
    run_migrations()
