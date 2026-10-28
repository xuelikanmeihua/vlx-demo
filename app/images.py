"""Decode caller-supplied images and JSON form fields.

Images are accepted as a multipart file upload or a base64 string only -
deliberately no fetch-by-URL option, to avoid giving this externally facing
API an SSRF vector.
"""
from __future__ import annotations

import base64
import binascii
import io
import json
from typing import List, Optional

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.config import settings


async def load_image(
    image: Optional[UploadFile], image_base64: Optional[str]
) -> Image.Image:
    if image is not None and image_base64:
        raise HTTPException(400, "Provide either `image` or `image_base64`, not both.")

    if image is not None:
        data = await _read_upload_capped(image)
    elif image_base64:
        data = _decode_base64(image_base64)
    else:
        raise HTTPException(400, "Provide an `image` file upload or an `image_base64` string.")

    if not data:
        raise HTTPException(400, "The image payload is empty.")
    if len(data) > settings.max_image_bytes:
        raise HTTPException(400, f"Image exceeds the {settings.max_image_bytes}-byte limit.")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(400, f"Could not decode image: {exc}") from exc


async def _read_upload_capped(image: UploadFile, chunk_size: int = 1024 * 1024) -> bytes:
    """Read an upload in chunks, aborting as soon as it exceeds the limit.

    `UploadFile.read()` with no size argument buffers the whole file before
    we get a chance to check it, which turns an oversized upload into an
    uncontrolled memory allocation. Reading in bounded chunks instead lets
    us reject early.
    """
    limit = settings.max_image_bytes
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await image.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(400, f"Image exceeds the {limit}-byte limit.")
        chunks.append(chunk)
    return b"".join(chunks)


def _decode_base64(value: str) -> bytes:
    if value.startswith("data:"):
        _, _, value = value.partition(",")
    # Base64 inflates size by ~4/3; reject obviously oversized payloads
    # before paying the cost of decoding them.
    if len(value) > (settings.max_image_bytes * 4 // 3) + 64:
        raise HTTPException(400, f"Image exceeds the {settings.max_image_bytes}-byte limit.")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(400, f"Invalid base64 image data: {exc}") from exc


def parse_json_box_list(value: Optional[str], field_name: str) -> Optional[List[List[float]]]:
    """Parse a JSON string of [[x1, y1, x2, y2], ...] pixel boxes.

    Box shape/range validation (exactly 4 numbers, positive area, clipped to
    the image) is left to VLXSeekWorker, which already performs it.
    """
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"`{field_name}` must be valid JSON: {exc}") from exc
    if not isinstance(parsed, list):
        raise HTTPException(400, f"`{field_name}` must be a JSON list of [x1, y1, x2, y2] boxes.")
    _check_max_items(parsed, field_name)
    return parsed


def parse_json_int_list(value: Optional[str], field_name: str) -> Optional[List[int]]:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"`{field_name}` must be valid JSON: {exc}") from exc
    if not isinstance(parsed, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in parsed
    ):
        raise HTTPException(400, f"`{field_name}` must be a JSON list of integers.")
    _check_max_items(parsed, field_name)
    return parsed


def _check_max_items(parsed: list, field_name: str) -> None:
    limit = settings.max_boxes_per_request
    if len(parsed) > limit:
        raise HTTPException(400, f"`{field_name}` must contain at most {limit} items.")
