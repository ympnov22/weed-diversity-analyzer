"""Simple API key authentication system."""

import secrets
from typing import Optional
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from ..database.models import UserModel

def create_api_key() -> str:
    """Create a new API key."""
    return secrets.token_urlsafe(32)

def verify_api_key(api_key: str, db: Session) -> Optional[UserModel]:
    """Verify an API key and return the associated user."""
    if not api_key:
        return None
    
    user = db.query(UserModel).filter(
        UserModel.api_key == api_key,
        UserModel.is_active == True
    ).first()
    
    return user

def get_current_user(api_key: str, db: Session) -> UserModel:
    """Get current user from API key."""
    user = verify_api_key(api_key, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
