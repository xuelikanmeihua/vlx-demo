"""Wires the vendored VLX-Seek checkout (third_party/VLX-Seek) onto sys.path.

VLX-Seek ships as a plain script repo (no setup.py/pyproject.toml): its own
Docker images just `COPY . .` and run `inference.py`/`vlx_seek_worker.py`
from the repo root. We vendor it as a git submodule and re-export the pieces
this service needs instead of duplicating VLX-Seek's model-loading and
region-proposal logic.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from app.config import settings

_VLX_SEEK_ROOT = Path(settings.vlx_seek_repo_path).resolve()


def _ensure_on_path() -> None:
    if not _VLX_SEEK_ROOT.is_dir():
        raise RuntimeError(
            f"VLX-Seek checkout not found at {_VLX_SEEK_ROOT}. Run "
            "`git submodule update --init --recursive`, or set "
            "VLX_API_VLX_SEEK_REPO_PATH to an existing checkout."
        )
    path_str = str(_VLX_SEEK_ROOT)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _load_cli_module():
    """Load VLX-Seek's inference.py under a private module name.

    Its top-level helpers (checkpoint download + WeDetect proposal
    generation) are only invoked from `if __name__ == "__main__"`, so
    importing the file as a library module is safe.
    """
    module_path = _VLX_SEEK_ROOT / "inference.py"
    spec = importlib.util.spec_from_file_location("_vlx_seek_cli", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"Could not load VLX-Seek CLI module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_ensure_on_path()
_cli = _load_cli_module()

from vlx_seek.hardware import resolve_device  # noqa: E402
from vlx_seek.task_templates import list_langs, list_tasks  # noqa: E402
from vlx_seek_worker import VLXSeekWorker  # noqa: E402

ensure_wedetect_checkpoint = _cli.ensure_wedetect_checkpoint
load_wedetect_proposals = _cli.load_wedetect_proposals
DEFAULT_DETECTOR_CHECKPOINT = _cli.DEFAULT_DETECTOR_CHECKPOINT

__all__ = [
    "resolve_device",
    "list_langs",
    "list_tasks",
    "VLXSeekWorker",
    "ensure_wedetect_checkpoint",
    "load_wedetect_proposals",
    "DEFAULT_DETECTOR_CHECKPOINT",
]
