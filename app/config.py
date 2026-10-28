"""Runtime configuration for the VLX-Seek FastAPI wrapper, via environment variables."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VLX_API_", env_file=".env", extra="ignore")

    # Hugging Face id or local checkpoint directory for VLX-Seek-1.5-10B.
    model_path: str = "omlab/VLX-Seek-1.5-10B"
    # Explicit device override (cuda:0, npu:0, mps, cpu). Leave unset to fall
    # back to VLX_SEEK_BACKEND / auto-detection, exactly like the upstream CLI.
    device: str | None = None

    vlx_seek_repo_path: str = str(REPO_ROOT / "third_party" / "VLX-Seek")
    resources_dir: str = str(REPO_ROOT / "resources")
    # WeDetect-Uni checkpoint used for automatic region proposals. Defaults to
    # <resources_dir>/wedetect_base_uni.pth; auto-downloaded on first use.
    detector_checkpoint: str | None = None

    # Serializes model.generate() calls; a single GPU-resident 10B model
    # should not be hit concurrently by default.
    max_concurrent_inference: int = 1
    max_image_bytes: int = 15 * 1024 * 1024
    # Rejected at the ASGI layer (via Content-Length) before the body is
    # read, so it must cover the image plus other multipart fields/overhead.
    max_request_bytes: int = 20 * 1024 * 1024
    # Upper bound for `max_new_tokens` on every generation endpoint, and for
    # the length of caller-supplied `bbox_list` / `target_region_indexes`.
    max_new_tokens_limit: int = 4096
    max_boxes_per_request: int = 200

    # When set, all /v1/* routes require header `X-API-Key: <api_key>`.
    api_key: str | None = None
    # Eagerly load the model during startup instead of on first request.
    load_model_on_startup: bool = False

    log_level: str = "info"
    host: str = "0.0.0.0"
    port: int = 8000

    @property
    def resolved_detector_checkpoint(self) -> str:
        return self.detector_checkpoint or str(
            Path(self.resources_dir) / "wedetect_base_uni.pth"
        )


settings = Settings()
