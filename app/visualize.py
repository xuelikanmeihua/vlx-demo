"""Render result_bbox_list onto the source image for API responses.

Mirrors inference.py's visualize_result_boxes, but returns PNG bytes/base64
instead of writing a file, since this service is stateless.
"""
from __future__ import annotations

import base64
import io
from typing import List

from PIL import Image, ImageDraw


def render_visualization_base64(image: Image.Image, result_bbox_list: List[dict]) -> str:
    visualized = image.copy()
    draw = ImageDraw.Draw(visualized)

    for box in result_bbox_list:
        xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
        label = box["label"]
        draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=3)
        try:
            draw.text((xmin + 2, ymin + 2), label, fill="red")
        except UnicodeEncodeError:
            draw.text((xmin + 2, ymin + 2), box["object_index"], fill="red")

    buffer = io.BytesIO()
    visualized.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
