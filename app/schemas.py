"""Response models for the VLX-Seek inference API."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ResultBBox(BaseModel):
    object_index: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    label: str


class InferenceResponse(BaseModel):
    answer: str
    result_bbox_list: List[ResultBBox] = Field(default_factory=list)
    prompt_tokens: int
    completion_tokens: int
    bbox_list: Optional[List[List[float]]] = None
    visualization_base64: Optional[str] = None


class TasksResponse(BaseModel):
    tasks: List[str]
    langs: List[str]


class ReadinessResponse(BaseModel):
    model_loaded: bool
    device: str
