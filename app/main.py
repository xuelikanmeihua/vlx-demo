"""FastAPI application entrypoint for the VLX-Seek inference service."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

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


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """Reject oversized requests via Content-Length, before the body is read.

    Without this, Starlette's multipart/form parser buffers the entire
    request body (file parts and plain text fields alike) into memory
    before application code gets a chance to check its size. This only
    catches requests that send a Content-Length header; chunked-encoded
    requests without one still fall through to the per-field checks in
    app.images, which bound how much *that* code buffers but not what the
    ASGI layer already read.
    """

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                size = int(content_length)
            except ValueError:
                size = None
            if size is not None and size > settings.max_request_bytes:
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": f"Request body exceeds the {settings.max_request_bytes}-byte limit."
                    },
                )
        return await call_next(request)


app.add_middleware(MaxBodySizeMiddleware)

app.include_router(health.router)
app.include_router(inference.router)
