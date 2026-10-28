"""Liveness/readiness endpoints. Intentionally unauthenticated for probes."""
from __future__ import annotations

from fastapi import APIRouter, Request

from app.model_manager import ModelManager
from app.schemas import ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@router.get("/readyz", response_model=ReadinessResponse)
async def readyz(request: Request) -> ReadinessResponse:
    manager: ModelManager = request.app.state.model_manager
    device = await manager.adevice()
    return ReadinessResponse(model_loaded=manager.is_model_loaded, device=device)
