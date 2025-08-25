"""FastAPI dependencies for authentication."""

from typing import Optional
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from ..database.database import get_db
from ..database.models import UserModel
from .auth import verify_api_key

async def get_api_key_from_header(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> Optional[str]:
    """Extract API key from Authorization header or X-API-Key header."""
    if authorization and authorization.startswith("Bearer "):
        return authorization[7:]
    elif x_api_key:
        return x_api_key
    return None

async def require_auth(
    api_key: Optional[str] = Depends(get_api_key_from_header),
    db: Session = Depends(get_db)
) -> UserModel:
    """Require authentication for protected endpoints."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = verify_api_key(api_key, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

async def optional_auth(
    api_key: Optional[str] = Depends(get_api_key_from_header),
    db: Session = Depends(get_db)
) -> Optional[UserModel]:
    """Optional authentication for endpoints that work with or without auth."""
    if not api_key:
        return None
    
    return verify_api_key(api_key, db)
