"""Inference endpoints wrapping the vendored VLX-Seek 1.5 worker.

Each route is a thin adapter over app.vendor.VLXSeekWorker: it decodes the
image, resolves candidate region boxes (explicit `bbox_list`, or WeDetect
auto-proposals), calls the worker method that matches the upstream CLI task,
and optionally renders a visualization of the selected regions.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from PIL import Image

from app.images import load_image, parse_json_box_list, parse_json_int_list
from app.model_manager import ModelManager
from app.schemas import InferenceResponse, TasksResponse
from app.security import require_api_key
from app.visualize import render_visualization_base64

router = APIRouter(prefix="/v1", tags=["inference"], dependencies=[Depends(require_api_key)])


def _manager(request: Request) -> ModelManager:
    return request.app.state.model_manager


async def _image_dep(
    image: Optional[UploadFile] = File(None, description="Image file upload."),
    image_base64: Optional[str] = Form(
        None, description="Base64-encoded image, alternative to `image`."
    ),
) -> Image.Image:
    return await load_image(image, image_base64)


async def _run(manager: ModelManager, fn: Callable[[Any], dict]) -> dict:
    try:
        return await manager.run(fn)
    except ValueError as exc:
        # VLXSeekWorker raises plain ValueError for bad boxes/indexes/args.
        raise HTTPException(400, str(exc)) from exc


async def _resolve_boxes(
    manager: ModelManager,
    image: Image.Image,
    bbox_list: Optional[str],
    auto_propose: bool,
) -> Optional[List[List[float]]]:
    boxes = parse_json_box_list(bbox_list, "bbox_list")
    if boxes is None and auto_propose:
        boxes = await manager.propose_boxes(image)
    return boxes


def _finalize(
    result: dict,
    image: Image.Image,
    boxes: Optional[List[List[float]]],
    visualize: bool,
) -> InferenceResponse:
    response = InferenceResponse(**result)
    if boxes is not None:
        response.bbox_list = boxes
    if visualize and result.get("result_bbox_list"):
        response.visualization_base64 = render_visualization_base64(
            image, result["result_bbox_list"]
        )
    return response


@router.get("/tasks", response_model=TasksResponse)
async def tasks() -> TasksResponse:
    from app.vendor import list_langs, list_tasks

    return TasksResponse(tasks=list_tasks(), langs=list_langs())


@router.post("/vqa", response_model=InferenceResponse)
async def vqa(
    request: Request,
    image: Image.Image = Depends(_image_dep),
    question: str = Form(..., description="Free-form question about the image."),
    max_new_tokens: int = Form(2048),
    temperature: float = Form(0.0),
    top_p: float = Form(1.0),
    repetition_penalty: float = Form(1.0),
) -> InferenceResponse:
    manager = _manager(request)
    result = await _run(
        manager,
        lambda worker: worker.predict(
            image,
            question,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        ),
    )
    return InferenceResponse(**result)


@router.post("/detect", response_model=InferenceResponse)
async def detect(
    request: Request,
    image: Image.Image = Depends(_image_dep),
    text: str = Form(
        ..., description='Category or categories, "; "-separated, e.g. "orange; apple".'
    ),
    lang: str = Form("en"),
    bbox_list: Optional[str] = Form(
        None, description="JSON [[x1,y1,x2,y2], ...] proposal boxes; auto-generated when omitted."
    ),
    auto_propose: bool = Form(True, description="Run WeDetect when `bbox_list` is omitted."),
    visualize: bool = Form(False),
    max_new_tokens: int = Form(2048),
    temperature: float = Form(0.0),
    top_p: float = Form(1.0),
    repetition_penalty: float = Form(1.0),
) -> InferenceResponse:
    manager = _manager(request)
    boxes = await _resolve_boxes(manager, image, bbox_list, auto_propose)
    if not boxes:
        raise HTTPException(400, "No proposal boxes available; pass `bbox_list` or enable `auto_propose`.")
    result = await _run(
        manager,
        lambda worker: worker.detect(
            image,
            boxes,
            text,
            lang=lang,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        ),
    )
    return _finalize(result, image, boxes, visualize)


@router.post("/ground", response_model=InferenceResponse)
async def ground(
    request: Request,
    image: Image.Image = Depends(_image_dep),
    description: str = Form(..., description="Referring expression identifying one instance."),
    lang: str = Form("en"),
    bbox_list: Optional[str] = Form(None),
    auto_propose: bool = Form(True),
    visualize: bool = Form(False),
    max_new_tokens: int = Form(2048),
    temperature: float = Form(0.0),
    top_p: float = Form(1.0),
    repetition_penalty: float = Form(1.0),
) -> InferenceResponse:
    manager = _manager(request)
    boxes = await _resolve_boxes(manager, image, bbox_list, auto_propose)
    if not boxes:
        raise HTTPException(400, "No proposal boxes available; pass `bbox_list` or enable `auto_propose`.")
    result = await _run(
        manager,
        lambda worker: worker.ground(
            image,
            boxes,
            description,
            lang=lang,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        ),
    )
    return _finalize(result, image, boxes, visualize)


@router.post("/count", response_model=InferenceResponse)
async def count(
    request: Request,
    image: Image.Image = Depends(_image_dep),
    description: str = Form(..., description="Target description to count, e.g. 'red cars'."),
    lang: str = Form("en"),
    bbox_list: Optional[str] = Form(None),
    auto_propose: bool = Form(True),
    visualize: bool = Form(False),
    max_new_tokens: int = Form(2048),
    temperature: float = Form(0.0),
    top_p: float = Form(1.0),
    repetition_penalty: float = Form(1.0),
) -> InferenceResponse:
    manager = _manager(request)
    boxes = await _resolve_boxes(manager, image, bbox_list, auto_propose)
    if not boxes:
        raise HTTPException(400, "No proposal boxes available; pass `bbox_list` or enable `auto_propose`.")
    result = await _run(
        manager,
        lambda worker: worker.count(
            image,
            boxes,
            description,
            lang=lang,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        ),
    )
    return _finalize(result, image, boxes, visualize)


@router.post("/reasoning-detect", response_model=InferenceResponse)
async def reasoning_detect(
    request: Request,
    image: Image.Image = Depends(_image_dep),
    text: str = Form(..., description="Target description or category to find with reasoning."),
    lang: str = Form("en"),
    bbox_list: Optional[str] = Form(None),
    auto_propose: bool = Form(True),
    visualize: bool = Form(False),
    max_new_tokens: int = Form(2048),
    temperature: float = Form(0.0),
    top_p: float = Form(1.0),
    repetition_penalty: float = Form(1.0),
) -> InferenceResponse:
    manager = _manager(request)
    boxes = await _resolve_boxes(manager, image, bbox_list, auto_propose)
    if not boxes:
        raise HTTPException(400, "No proposal boxes available; pass `bbox_list` or enable `auto_propose`.")
    result = await _run(
        manager,
        lambda worker: worker.reasoning_detect(
            image,
            boxes,
            text,
            lang=lang,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        ),
    )
    return _finalize(result, image, boxes, visualize)


@router.post("/region/caption", response_model=InferenceResponse)
async def region_caption(
    request: Request,
    image: Image.Image = Depends(_image_dep),
    bbox_list: str = Form(..., description="JSON [[x1,y1,x2,y2], ...] candidate boxes."),
    target_region_indexes: str = Form(
        ..., description="JSON list of indexes into `bbox_list`, e.g. '[0, 2]'."
    ),
    lang: str = Form("en"),
    visualize: bool = Form(False),
    max_new_tokens: int = Form(2048),
    temperature: float = Form(0.0),
    top_p: float = Form(1.0),
    repetition_penalty: float = Form(1.0),
) -> InferenceResponse:
    manager = _manager(request)
    boxes = parse_json_box_list(bbox_list, "bbox_list")
    indexes = parse_json_int_list(target_region_indexes, "target_region_indexes")
    result = await _run(
        manager,
        lambda worker: worker.describe_region(
            image,
            boxes,
            indexes,
            lang=lang,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        ),
    )
    return _finalize(result, image, boxes, visualize)


@router.post("/region/ocr", response_model=InferenceResponse)
async def region_ocr(
    request: Request,
    image: Image.Image = Depends(_image_dep),
    bbox_list: str = Form(..., description="JSON [[x1,y1,x2,y2], ...] candidate boxes."),
    target_region_indexes: str = Form(
        ..., description="JSON list of indexes into `bbox_list`, e.g. '[0, 2]'."
    ),
    lang: str = Form("en"),
    visualize: bool = Form(False),
    max_new_tokens: int = Form(2048),
    temperature: float = Form(0.0),
    top_p: float = Form(1.0),
    repetition_penalty: float = Form(1.0),
) -> InferenceResponse:
    manager = _manager(request)
    boxes = parse_json_box_list(bbox_list, "bbox_list")
    indexes = parse_json_int_list(target_region_indexes, "target_region_indexes")
    result = await _run(
        manager,
        lambda worker: worker.read_region_text(
            image,
            boxes,
            indexes,
            lang=lang,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        ),
    )
    return _finalize(result, image, boxes, visualize)


@router.post("/predict", response_model=InferenceResponse)
async def predict(
    request: Request,
    image: Image.Image = Depends(_image_dep),
    prompt: str = Form(..., description="Raw prompt sent to the model."),
    bbox_list: Optional[str] = Form(
        None, description="Optional JSON candidate boxes for region-aware prompts."
    ),
    visualize: bool = Form(False),
    max_new_tokens: int = Form(2048),
    temperature: float = Form(0.0),
    top_p: float = Form(1.0),
    repetition_penalty: float = Form(1.0),
) -> InferenceResponse:
    manager = _manager(request)
    boxes = parse_json_box_list(bbox_list, "bbox_list")
    result = await _run(
        manager,
        lambda worker: worker.predict(
            image,
            prompt,
            bbox_list=boxes,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        ),
    )
    return _finalize(result, image, boxes, visualize)
