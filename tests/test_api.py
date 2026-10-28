"""API-layer smoke tests.

These exercise routing, image decoding, request validation, auth, and
response shaping *without* loading the real 10B VLX-Seek model: ModelManager
is monkeypatched with a FakeWorker that mimics VLXSeekWorker's return shape.
"""
from __future__ import annotations

import base64
import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


def _sample_image_bytes() -> bytes:
    image = Image.new("RGB", (32, 32), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FakeWorker:
    def predict(self, image, question, **kwargs) -> dict:
        return {
            "answer": f"fake-answer:{question}",
            "result_bbox_list": [],
            "prompt_tokens": 10,
            "completion_tokens": 3,
        }

    def _grounded(self, label: str) -> dict:
        return {
            "answer": f"<ground>{label}</ground><objects><obj0></objects>",
            "result_bbox_list": [
                {
                    "object_index": "<obj0>",
                    "xmin": 1.0,
                    "ymin": 2.0,
                    "xmax": 10.0,
                    "ymax": 12.0,
                    "label": label,
                }
            ],
            "prompt_tokens": 20,
            "completion_tokens": 8,
        }

    def detect(self, image, boxes, text, **kwargs) -> dict:
        return self._grounded(str(text))

    def ground(self, image, boxes, description, **kwargs) -> dict:
        return self._grounded(description)

    def count(self, image, boxes, description, **kwargs) -> dict:
        return self._grounded(description)

    def reasoning_detect(self, image, boxes, text, **kwargs) -> dict:
        return self._grounded(str(text))

    def describe_region(self, image, boxes, indexes, **kwargs) -> dict:
        return {
            "answer": f"region {indexes} looks like a box.",
            "result_bbox_list": [],
            "prompt_tokens": 15,
            "completion_tokens": 6,
        }

    def read_region_text(self, image, boxes, indexes, **kwargs) -> dict:
        return {
            "answer": f"OCR:{indexes}",
            "result_bbox_list": [],
            "prompt_tokens": 15,
            "completion_tokens": 6,
        }


async def _fake_propose_boxes(image) -> list:
    return [[0.0, 0.0, 16.0, 16.0], [16.0, 16.0, 32.0, 32.0]]


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        manager = app.state.model_manager
        manager._worker = FakeWorker()
        manager.ensure_loaded = lambda: None
        manager.propose_boxes = _fake_propose_boxes
        yield test_client


def test_healthz(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readyz_reports_loaded_model(client: TestClient) -> None:
    resp = client.get("/readyz")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


def test_vqa_with_file_upload(client: TestClient) -> None:
    resp = client.post(
        "/v1/vqa",
        data={"question": "What fruits are in the image?"},
        files={"image": ("test.png", _sample_image_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "fake-answer:What fruits are in the image?"
    assert body["result_bbox_list"] == []


def test_vqa_with_base64_image(client: TestClient) -> None:
    encoded = base64.b64encode(_sample_image_bytes()).decode("ascii")
    resp = client.post(
        "/v1/vqa",
        data={"question": "describe", "image_base64": encoded},
    )
    assert resp.status_code == 200


def test_vqa_rejects_both_image_sources(client: TestClient) -> None:
    encoded = base64.b64encode(_sample_image_bytes()).decode("ascii")
    resp = client.post(
        "/v1/vqa",
        data={"question": "describe", "image_base64": encoded},
        files={"image": ("test.png", _sample_image_bytes(), "image/png")},
    )
    assert resp.status_code == 400


def test_vqa_rejects_invalid_base64(client: TestClient) -> None:
    resp = client.post(
        "/v1/vqa",
        data={"question": "describe", "image_base64": "not-valid-base64!!"},
    )
    assert resp.status_code == 400


def test_detect_with_explicit_bbox_list(client: TestClient) -> None:
    resp = client.post(
        "/v1/detect",
        data={
            "text": "orange; apple",
            "bbox_list": "[[1, 2, 10, 12]]",
        },
        files={"image": ("test.png", _sample_image_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["result_bbox_list"][0]["label"] == "orange; apple"
    assert body["bbox_list"] == [[1, 2, 10, 12]]


def test_detect_auto_propose(client: TestClient) -> None:
    resp = client.post(
        "/v1/detect",
        data={"text": "orange", "auto_propose": "true"},
        files={"image": ("test.png", _sample_image_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["bbox_list"] == [[0.0, 0.0, 16.0, 16.0], [16.0, 16.0, 32.0, 32.0]]


def test_detect_without_boxes_or_auto_propose_fails(client: TestClient) -> None:
    resp = client.post(
        "/v1/detect",
        data={"text": "orange", "auto_propose": "false"},
        files={"image": ("test.png", _sample_image_bytes(), "image/png")},
    )
    assert resp.status_code == 400


def test_detect_with_visualization(client: TestClient) -> None:
    resp = client.post(
        "/v1/detect",
        data={"text": "orange", "bbox_list": "[[1, 2, 10, 12]]", "visualize": "true"},
        files={"image": ("test.png", _sample_image_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["visualization_base64"] is not None


def test_region_caption_requires_indexes(client: TestClient) -> None:
    resp = client.post(
        "/v1/region/caption",
        data={"bbox_list": "[[1, 2, 10, 12]]"},
        files={"image": ("test.png", _sample_image_bytes(), "image/png")},
    )
    assert resp.status_code == 422


def test_region_caption_success(client: TestClient) -> None:
    resp = client.post(
        "/v1/region/caption",
        data={"bbox_list": "[[1, 2, 10, 12]]", "target_region_indexes": "[0]"},
        files={"image": ("test.png", _sample_image_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    assert "[0]" in resp.json()["answer"]


def test_tasks_endpoint(client: TestClient) -> None:
    resp = client.get("/v1/tasks")
    assert resp.status_code == 200
    body = resp.json()
    assert "detection" in body["tasks"]
    assert "en" in body["langs"]


def test_ground_success(client: TestClient) -> None:
    resp = client.post(
        "/v1/ground",
        data={"description": "the red apple", "bbox_list": "[[1, 2, 10, 12]]"},
        files={"image": ("test.png", _sample_image_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["result_bbox_list"][0]["label"] == "the red apple"


def test_count_success(client: TestClient) -> None:
    resp = client.post(
        "/v1/count",
        data={"description": "apples", "bbox_list": "[[1, 2, 10, 12]]"},
        files={"image": ("test.png", _sample_image_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["result_bbox_list"][0]["label"] == "apples"


def test_reasoning_detect_success(client: TestClient) -> None:
    resp = client.post(
        "/v1/reasoning-detect",
        data={"text": "a ripe fruit", "bbox_list": "[[1, 2, 10, 12]]"},
        files={"image": ("test.png", _sample_image_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["result_bbox_list"][0]["label"] == "a ripe fruit"


def test_region_ocr_success(client: TestClient) -> None:
    resp = client.post(
        "/v1/region/ocr",
        data={"bbox_list": "[[1, 2, 10, 12]]", "target_region_indexes": "[0]"},
        files={"image": ("test.png", _sample_image_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    assert "[0]" in resp.json()["answer"]


def test_predict_raw_prompt(client: TestClient) -> None:
    resp = client.post(
        "/v1/predict",
        data={"prompt": "Describe this image."},
        files={"image": ("test.png", _sample_image_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["answer"] == "fake-answer:Describe this image."


def test_worker_value_error_is_mapped_to_400(client: TestClient) -> None:
    def _raise(*args, **kwargs):
        raise ValueError("bbox must have positive area after clipping")

    app.state.model_manager._worker.detect = _raise

    resp = client.post(
        "/v1/detect",
        data={"text": "orange", "bbox_list": "[[1, 1, 1, 1]]"},
        files={"image": ("test.png", _sample_image_bytes(), "image/png")},
    )
    assert resp.status_code == 400
    assert "positive area" in resp.json()["detail"]


def test_vqa_rejects_max_new_tokens_above_limit(client: TestClient) -> None:
    resp = client.post(
        "/v1/vqa",
        data={"question": "describe", "max_new_tokens": "999999"},
        files={"image": ("test.png", _sample_image_bytes(), "image/png")},
    )
    assert resp.status_code == 422


def test_detect_rejects_too_many_boxes(client: TestClient) -> None:
    from app.config import settings

    too_many = [[0, 0, 1, 1]] * (settings.max_boxes_per_request + 1)
    resp = client.post(
        "/v1/detect",
        data={"text": "orange", "bbox_list": json.dumps(too_many)},
        files={"image": ("test.png", _sample_image_bytes(), "image/png")},
    )
    assert resp.status_code == 400


def test_vqa_rejects_oversized_image(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app import images

    monkeypatch.setattr(images.settings, "max_image_bytes", 10)

    resp = client.post(
        "/v1/vqa",
        data={"question": "describe"},
        files={"image": ("test.png", _sample_image_bytes(), "image/png")},
    )
    assert resp.status_code == 400


def test_api_key_enforced_when_configured(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from app import security

    monkeypatch.setattr(security.settings, "api_key", "secret-key")

    resp = client.get("/v1/tasks")
    assert resp.status_code == 401

    resp = client.get("/v1/tasks", headers={"X-API-Key": "secret-key"})
    assert resp.status_code == 200

    resp = client.get("/healthz")
    assert resp.status_code == 200
