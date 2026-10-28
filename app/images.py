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
        data = await image.read()
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


def _decode_base64(value: str) -> bytes:
    if value.startswith("data:"):
        _, _, value = value.partition(",")
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
    return parsed
