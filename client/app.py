"""Local Gradio test client for the deployed VLX-Seek inference API.

Pure HTTP client -- no torch/transformers/GPU needed. Run this on a laptop
or dev box to try each endpoint with real images against a running
deployment and see request latency + the returned result image.

Usage:
    pip install -r client/requirements.txt
    python client/app.py
"""
from __future__ import annotations

import base64
import io
import json
import time
from typing import Any, Optional

import gradio as gr
import requests
from PIL import Image

DEFAULT_BASE_URL = "http://117.50.199.161:8000"


# --------------------------------------------------------------------------
# HTTP helpers
# --------------------------------------------------------------------------

def _image_file_tuple(image: Image.Image) -> tuple[str, bytes, str]:
    # JPEG, not PNG: PNG is lossless and can easily blow past the server's
    # 15MB image / 20MB request-body limits for an ordinary photo, where
    # JPEG at quality 90 stays a fraction of the size for the same content.
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return ("upload.jpg", buf.getvalue(), "image/jpeg")


def _call_api(
    base_url: str,
    api_key: str,
    method: str,
    path: str,
    data: Optional[dict] = None,
    files: Optional[dict] = None,
) -> tuple[float, dict[str, Any]]:
    """Call the API and print+return the elapsed wall-clock time.

    Always returns (elapsed_seconds, payload); payload has an "error" key
    on any failure (network error, non-2xx status, bad JSON) so callers
    don't need their own try/except.
    """
    url = base_url.rstrip("/") + path
    headers = {"X-API-Key": api_key} if api_key else {}
    start = time.perf_counter()
    try:
        if method == "GET":
            resp = requests.get(url, headers=headers, timeout=120)
        else:
            resp = requests.post(url, headers=headers, data=data, files=files, timeout=120)
    except requests.RequestException as exc:
        elapsed = time.perf_counter() - start
        print(f"[{path}] 请求失败,耗时 {elapsed:.3f}s: {exc}")
        return elapsed, {"error": str(exc)}

    elapsed = time.perf_counter() - start
    print(f"[{path}] 耗时 {elapsed:.3f}s,状态码 {resp.status_code}")

    try:
        payload = resp.json()
    except ValueError:
        return elapsed, {"error": f"HTTP {resp.status_code}: {resp.text[:500]}"}
    if resp.status_code >= 400:
        return elapsed, {"error": payload.get("detail", f"HTTP {resp.status_code}")}
    return elapsed, payload


def _decode_image(b64: Optional[str]) -> Optional[Image.Image]:
    if not b64:
        return None
    try:
        return Image.open(io.BytesIO(base64.b64decode(b64)))
    except Exception:
        return None


def _status_line(elapsed: float, payload: dict[str, Any]) -> str:
    if "error" in payload:
        return f"❌ 耗时 {elapsed:.3f}s —— 出错: {payload['error']}"
    return f"✅ 耗时 {elapsed:.3f}s"


def _boxes_json(payload: dict[str, Any]) -> Any:
    return payload.get("result_bbox_list", []) if "error" not in payload else {"error": payload["error"]}


def _result_image(payload: dict[str, Any], fallback: Optional[Image.Image]) -> Optional[Image.Image]:
    return _decode_image(payload.get("visualization_base64")) or fallback


# --------------------------------------------------------------------------
# Shared request bodies
# --------------------------------------------------------------------------

def _gen_params(max_new_tokens: float, temperature: float, top_p: float, repetition_penalty: float) -> dict:
    return {
        "max_new_tokens": int(max_new_tokens),
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
    }


def _gen_params_accordion() -> tuple[gr.Slider, gr.Slider, gr.Slider, gr.Slider]:
    with gr.Accordion("生成参数(可选)", open=False):
        max_new_tokens = gr.Slider(1, 4096, value=512, step=1, label="max_new_tokens")
        temperature = gr.Slider(0.0, 2.0, value=0.0, step=0.1, label="temperature")
        top_p = gr.Slider(0.01, 1.0, value=1.0, step=0.01, label="top_p")
        repetition_penalty = gr.Slider(0.01, 2.0, value=1.0, step=0.1, label="repetition_penalty")
    return max_new_tokens, temperature, top_p, repetition_penalty


# --------------------------------------------------------------------------
# Per-task call functions (one per distinct request shape; endpoints that
# share a shape share a function via a bound `path`/`field_name`).
# --------------------------------------------------------------------------

def call_vqa(base_url, api_key, image, question, max_new_tokens, temperature, top_p, repetition_penalty):
    if image is None:
        return "请先上传图片", "", [], None
    data = {"question": question, **_gen_params(max_new_tokens, temperature, top_p, repetition_penalty)}
    elapsed, payload = _call_api(
        base_url, api_key, "POST", "/v1/vqa", data=data, files={"image": _image_file_tuple(image)}
    )
    if "error" in payload:
        return _status_line(elapsed, payload), "", [], image
    return _status_line(elapsed, payload), payload.get("answer", ""), _boxes_json(payload), image


def call_proposal_task(
    path: str,
    field_name: str,
    base_url, api_key, image, text_value, lang, bbox_list_text, auto_propose, visualize,
    max_new_tokens, temperature, top_p, repetition_penalty,
):
    """Shared body for /v1/detect, /v1/ground, /v1/count, /v1/reasoning-detect."""
    if image is None:
        return "请先上传图片", "", [], None
    data = {
        field_name: text_value,
        "lang": lang,
        "auto_propose": str(bool(auto_propose)).lower(),
        "visualize": str(bool(visualize)).lower(),
        **_gen_params(max_new_tokens, temperature, top_p, repetition_penalty),
    }
    bbox_list_text = (bbox_list_text or "").strip()
    if bbox_list_text:
        data["bbox_list"] = bbox_list_text
    elapsed, payload = _call_api(base_url, api_key, "POST", path, data=data, files={"image": _image_file_tuple(image)})
    if "error" in payload:
        return _status_line(elapsed, payload), "", [], image
    return _status_line(elapsed, payload), payload.get("answer", ""), _boxes_json(payload), _result_image(payload, image)


def call_region_task(
    path: str,
    base_url, api_key, image, bbox_list_text, indexes_text, lang, visualize,
    max_new_tokens, temperature, top_p, repetition_penalty,
):
    """Shared body for /v1/region/caption and /v1/region/ocr."""
    if image is None:
        return "请先上传图片", "", [], None
    data = {
        "bbox_list": bbox_list_text,
        "target_region_indexes": indexes_text,
        "lang": lang,
        "visualize": str(bool(visualize)).lower(),
        **_gen_params(max_new_tokens, temperature, top_p, repetition_penalty),
    }
    elapsed, payload = _call_api(base_url, api_key, "POST", path, data=data, files={"image": _image_file_tuple(image)})
    if "error" in payload:
        return _status_line(elapsed, payload), "", [], image
    return _status_line(elapsed, payload), payload.get("answer", ""), _boxes_json(payload), _result_image(payload, image)


def call_predict(base_url, api_key, image, prompt, bbox_list_text, visualize, max_new_tokens, temperature, top_p, repetition_penalty):
    if image is None:
        return "请先上传图片", "", [], None
    data = {
        "prompt": prompt,
        "visualize": str(bool(visualize)).lower(),
        **_gen_params(max_new_tokens, temperature, top_p, repetition_penalty),
    }
    bbox_list_text = (bbox_list_text or "").strip()
    if bbox_list_text:
        data["bbox_list"] = bbox_list_text
    elapsed, payload = _call_api(base_url, api_key, "POST", "/v1/predict", data=data, files={"image": _image_file_tuple(image)})
    if "error" in payload:
        return _status_line(elapsed, payload), "", [], image
    return _status_line(elapsed, payload), payload.get("answer", ""), _boxes_json(payload), _result_image(payload, image)


def call_tasks(base_url, api_key):
    elapsed, payload = _call_api(base_url, api_key, "GET", "/v1/tasks")
    return _status_line(elapsed, payload), payload


def call_health(base_url, api_key):
    elapsed, payload = _call_api(base_url, api_key, "GET", "/healthz")
    return _status_line(elapsed, payload), payload


def call_ready(base_url, api_key):
    elapsed, payload = _call_api(base_url, api_key, "GET", "/readyz")
    return _status_line(elapsed, payload), payload


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

def _proposal_tab(name: str, path: str, field_name: str, field_label: str, base_url, api_key):
    with gr.Tab(name):
        image = gr.Image(type="pil", label="上传图片")
        text_value = gr.Textbox(label=field_label)
        with gr.Row():
            lang = gr.Dropdown(["en", "zh"], value="en", label="lang")
            auto_propose = gr.Checkbox(value=True, label="auto_propose(自动生成候选框)")
            visualize = gr.Checkbox(value=True, label="visualize(返回画框结果图)")
        bbox_list_text = gr.Textbox(label="bbox_list(可选,JSON,留空则按上面的 auto_propose 处理)", placeholder="[[12,20,180,210]]")
        gen = _gen_params_accordion()
        btn = gr.Button(f"调用 {path}", variant="primary")
        status = gr.Textbox(label="状态/耗时", interactive=False)
        answer = gr.Textbox(label="answer", interactive=False)
        boxes = gr.JSON(label="result_bbox_list")
        result_image = gr.Image(type="pil", label="结果图(若有 visualize 结果则画框,否则显示原图)")

        btn.click(
            lambda *args: call_proposal_task(path, field_name, *args),
            inputs=[base_url, api_key, image, text_value, lang, bbox_list_text, auto_propose, visualize, *gen],
            outputs=[status, answer, boxes, result_image],
        )


def _region_tab(name: str, path: str, base_url, api_key):
    with gr.Tab(name):
        image = gr.Image(type="pil", label="上传图片")
        bbox_list_text = gr.Textbox(label="bbox_list(必填,JSON)", placeholder="[[12,20,180,210],[50,60,220,300]]")
        indexes_text = gr.Textbox(label="target_region_indexes(必填,JSON,下标数组)", placeholder="[0]")
        with gr.Row():
            lang = gr.Dropdown(["en", "zh"], value="en", label="lang")
            visualize = gr.Checkbox(value=False, label="visualize(该任务通常不产生新框,一般没有结果图)")
        gen = _gen_params_accordion()
        btn = gr.Button(f"调用 {path}", variant="primary")
        status = gr.Textbox(label="状态/耗时", interactive=False)
        answer = gr.Textbox(label="answer", interactive=False)
        boxes = gr.JSON(label="result_bbox_list")
        result_image = gr.Image(type="pil", label="结果图")

        btn.click(
            lambda *args: call_region_task(path, *args),
            inputs=[base_url, api_key, image, bbox_list_text, indexes_text, lang, visualize, *gen],
            outputs=[status, answer, boxes, result_image],
        )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="VLX-Seek API 测试客户端") as demo:
        gr.Markdown(
            "## VLX-Seek 推理服务 API 测试客户端\n"
            "本程序只通过 HTTP 调用已部署的服务,不在本机做推理,因此不需要 GPU/torch。"
            "每次调用的耗时会打印在运行本程序的终端里,也会显示在下面的状态框中。"
        )
        with gr.Row():
            base_url = gr.Textbox(label="Base URL", value=DEFAULT_BASE_URL, scale=3)
            api_key = gr.Textbox(label="API Key", type="password", scale=2)

        with gr.Tab("VQA 问答"):
            image = gr.Image(type="pil", label="上传图片")
            question = gr.Textbox(label="question")
            gen = _gen_params_accordion()
            btn = gr.Button("调用 /v1/vqa", variant="primary")
            status = gr.Textbox(label="状态/耗时", interactive=False)
            answer = gr.Textbox(label="answer", interactive=False)
            boxes = gr.JSON(label="result_bbox_list")
            result_image = gr.Image(type="pil", label="图片")
            btn.click(
                call_vqa,
                inputs=[base_url, api_key, image, question, *gen],
                outputs=[status, answer, boxes, result_image],
            )

        _proposal_tab("Detect 检测", "/v1/detect", "text", 'text(类别,多个用 "; " 分隔,如 "orange; apple")', base_url, api_key)
        _proposal_tab("Ground 指代定位", "/v1/ground", "description", "description(唯一目标的指代表达)", base_url, api_key)
        _proposal_tab("Count 计数", "/v1/count", "description", "description(要计数的目标描述)", base_url, api_key)
        _proposal_tab("Reasoning-Detect 推理检测", "/v1/reasoning-detect", "text", "text(需要推理才能定位的描述)", base_url, api_key)

        _region_tab("Region Caption 区域简述", "/v1/region/caption", base_url, api_key)
        _region_tab("Region OCR 区域文字识别", "/v1/region/ocr", base_url, api_key)

        with gr.Tab("Predict 原始 Prompt"):
            image = gr.Image(type="pil", label="上传图片")
            prompt = gr.Textbox(label="prompt")
            bbox_list_text = gr.Textbox(label="bbox_list(可选,JSON)", placeholder="[[12,20,180,210]]")
            visualize = gr.Checkbox(value=True, label="visualize")
            gen = _gen_params_accordion()
            btn = gr.Button("调用 /v1/predict", variant="primary")
            status = gr.Textbox(label="状态/耗时", interactive=False)
            answer = gr.Textbox(label="answer", interactive=False)
            boxes = gr.JSON(label="result_bbox_list")
            result_image = gr.Image(type="pil", label="结果图")
            btn.click(
                call_predict,
                inputs=[base_url, api_key, image, prompt, bbox_list_text, visualize, *gen],
                outputs=[status, answer, boxes, result_image],
            )

        with gr.Tab("Tasks / 健康检查"):
            tasks_btn = gr.Button("GET /v1/tasks")
            tasks_status = gr.Textbox(label="状态/耗时", interactive=False)
            tasks_json = gr.JSON(label="响应")
            tasks_btn.click(call_tasks, inputs=[base_url, api_key], outputs=[tasks_status, tasks_json])

            health_btn = gr.Button("GET /healthz(无需 API Key)")
            health_status = gr.Textbox(label="状态/耗时", interactive=False)
            health_json = gr.JSON(label="响应")
            health_btn.click(call_health, inputs=[base_url, api_key], outputs=[health_status, health_json])

            ready_btn = gr.Button("GET /readyz(无需 API Key)")
            ready_status = gr.Textbox(label="状态/耗时", interactive=False)
            ready_json = gr.JSON(label="响应")
            ready_btn.click(call_ready, inputs=[base_url, api_key], outputs=[ready_status, ready_json])

    return demo


if __name__ == "__main__":
    build_app().launch()
