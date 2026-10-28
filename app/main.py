"""FastAPI application entrypoint for the VLX-Seek inference service."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.model_manager import ModelManager
from app.routers import health, inference

logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger("vlx_demo")


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager = ModelManager(settings)
    app.state.model_manager = manager
    if settings.load_model_on_startup:
        logger.info("Loading VLX-Seek model at startup ...")
        await manager.aensure_loaded()
    yield
    manager.shutdown()


app = FastAPI(
    title="VLX-Seek Inference API",
    description=(
        "FastAPI wrapper exposing om-ai-lab/VLX-Seek 1.5 "
        "(fine-grained perception VLM) for detection, grounding, counting, "
        "region captioning/OCR, and general VQA."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error while processing %s %s", request.method, request.url)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


app.include_router(health.router)
app.include_router(inference.router)
