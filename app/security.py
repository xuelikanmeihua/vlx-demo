"""Optional API-key gate for externally exposed /v1/* routes."""
from __future__ import annotations

import secrets
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config import settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: Optional[str] = Security(_api_key_header)) -> None:
    if not settings.api_key:
        # No key configured: auth is disabled (e.g. local development).
        return
    if api_key is None or not secrets.compare_digest(api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
