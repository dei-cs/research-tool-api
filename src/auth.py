from fastapi import Security, HTTPException, status, Request
from fastapi.security import APIKeyHeader
from .app_settings import settings

# Define API Key header for frontend authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_frontend_api_key(
    request: Request,
    api_key: str = Security(api_key_header)
) -> str:
    """
    Verify the API key provided by the frontend.
    Supports both X-API-Key header and Authorization Bearer token.
    
    Args:
        request: FastAPI request object
        api_key: API key from the X-API-Key header
        
    Returns:
        The validated API key
        
    Raises:
        HTTPException: If API key is invalid
    """
    # Try X-API-Key header first
    if api_key and api_key == settings.frontend_api_key:
        return api_key
    
    # Try Authorization Bearer token
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]  # Remove "Bearer " prefix
        if token == settings.frontend_api_key:
            return token
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key"
    )


async def verify_frontend_api_key_optional(request: Request) -> bool:
    """
    Optional authentication check - returns True if authenticated, False otherwise.
    Does not raise exception.
    """
    try:
        await verify_frontend_api_key(request)
        return True
    except HTTPException:
        return False
