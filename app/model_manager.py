"""Loads and serves the VLX-Seek worker and region-proposal detector.

The VLM is loaded lazily (on first request, or at startup when
VLX_API_LOAD_MODEL_ON_STARTUP is set) and kept resident for the process
lifetime. All blocking calls run on a small dedicated thread pool so the
asyncio event loop stays responsive; the pool size also doubles as the
concurrency limit for a single GPU-resident model.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List

from PIL import Image

from app.config import Settings

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._worker: Any = None
        self._device: Any = None
        self._load_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, settings.max_concurrent_inference),
            thread_name_prefix="vlx-seek-infer",
        )

    @property
    def is_model_loaded(self) -> bool:
        return self._worker is not None

    @property
    def device(self) -> str:
        self._ensure_device_resolved()
        return str(self._device)

    def _ensure_device_resolved(self) -> None:
        if self._device is not None:
            return
        from app.vendor import resolve_device

        self._device = resolve_device(self._settings.device)

    def ensure_loaded(self) -> None:
        """Load the VLX-Seek worker if needed. Blocking; call off the event loop."""
        if self._worker is not None:
            return
        with self._load_lock:
            if self._worker is not None:
                return
            self._ensure_device_resolved()
            from app.vendor import VLXSeekWorker

            logger.info(
                "Loading VLX-Seek model %r on %s ...",
                self._settings.model_path,
                self._device,
            )
            self._worker = VLXSeekWorker(self._settings.model_path, device=self._device)
            logger.info("VLX-Seek model ready.")

    async def aensure_loaded(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self.ensure_loaded)

    async def run(self, fn: Callable[[Any], dict]) -> dict:
        """Run a blocking worker call on the inference executor."""
        loop = asyncio.get_running_loop()

        def _call() -> dict:
            self.ensure_loaded()
            return fn(self._worker)

        return await loop.run_in_executor(self._executor, _call)

    async def propose_boxes(self, image: Image.Image) -> List[List[float]]:
        """Generate WeDetect-Uni region proposals for ``image`` (pixel xyxy)."""
        self._ensure_device_resolved()
        from app.vendor import load_wedetect_proposals

        loop = asyncio.get_running_loop()

        def _call() -> List[List[float]]:
            return load_wedetect_proposals(
                image, self._settings.resolved_detector_checkpoint, device=self._device
            )

        return await loop.run_in_executor(self._executor, _call)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
