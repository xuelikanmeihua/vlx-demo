# VLX-Seek Inference API

A FastAPI service that exposes [om-ai-lab/VLX-Seek](https://github.com/om-ai-lab/VLX-Seek)
1.5-10B — a fine-grained perception VLM that grounds language to image
regions instead of generating raw coordinates — as an external HTTP API.

This repo vendors VLX-Seek as a git submodule (`third_party/VLX-Seek`, no
upstream fork or code duplication) and wraps its `VLXSeekWorker` Python API
in a small, stateless FastAPI layer: image decoding, request validation,
optional region-proposal generation, optional API-key auth, and JSON
responses.

## Architecture

```
app/
├── main.py            FastAPI app, lifespan, body-size middleware, global error handler
├── config.py           Settings from environment variables (VLX_API_*)
├── vendor.py           Wires third_party/VLX-Seek onto sys.path; re-exports its API
├── model_manager.py     Lazily loads VLXSeekWorker; runs blocking inference
│                        on a dedicated thread pool (also the concurrency limit)
├── images.py           Decodes uploaded/base64 images; parses JSON form fields
├── visualize.py         Renders result boxes to a PNG, returned as base64
├── security.py          Optional `X-API-Key` gate for /v1/* routes
├── schemas.py           Response models
└── routers/
    ├── health.py        /healthz, /readyz (unauthenticated)
    └── inference.py      /v1/* task endpoints
```

`app/vendor.py` is the only integration seam: it adds `third_party/VLX-Seek`
to `sys.path`, loads its `inference.py` as a private module (for the WeDetect
proposal-download/generation helpers), and re-exports `VLXSeekWorker`,
`resolve_device`, and the task-template helpers. No VLX-Seek source is
copied or modified.

## Endpoints

All `/v1/*` endpoints accept `multipart/form-data`: an image (`image` file
upload, or `image_base64` string) plus task-specific fields. They mirror
VLX-Seek's own task set 1:1 — see [VLX-Seek's inference guide](third_party/VLX-Seek/docs/inference.md)
for task semantics.

| Method | Path | VLX-Seek task | Notes |
| --- | --- | --- | --- |
| POST | `/v1/vqa` | `vqa` | `question`; no region proposals |
| POST | `/v1/detect` | `detection` | `text` (`"orange; apple"`), `bbox_list?`, `auto_propose` |
| POST | `/v1/ground` | `grounding_single` | `description`, `bbox_list?`, `auto_propose` |
| POST | `/v1/count` | `counting` | `description`, `bbox_list?`, `auto_propose` |
| POST | `/v1/reasoning-detect` | `reasoning_detection` | `text`, `bbox_list?`, `auto_propose` |
| POST | `/v1/region/caption` | `brief_region_caption` | `bbox_list`, `target_region_indexes` (both required) |
| POST | `/v1/region/ocr` | `region_ocr` | `bbox_list`, `target_region_indexes` (both required) |
| POST | `/v1/predict` | raw prompt | `prompt`, optional `bbox_list` — direct `VLXSeekWorker.predict` passthrough |
| GET | `/v1/tasks` | — | lists supported task/lang names |
| GET | `/healthz` / `/readyz` | — | liveness / model-loaded + device, unauthenticated |

Common optional fields on generation endpoints: `lang` (`en`/`zh`),
`max_new_tokens` (capped at `VLX_API_MAX_NEW_TOKENS_LIMIT`), `temperature`,
`top_p`, `repetition_penalty`, and `visualize` (returns
`visualization_base64`, a PNG with result boxes drawn). These four
generation fields, plus the box-resolution/visualization logic shared by
the detection-family and region endpoints, live in one place
(`_generation_params`/`_proposal_task`/`_region_task` in
`app/routers/inference.py`) rather than being repeated per route.

When `bbox_list` is omitted on a detection-family endpoint and
`auto_propose` is true (default), the service runs VLX-Seek's bundled
WeDetect-Uni detector to generate proposals, exactly like the upstream CLI
when `--bbox-list` is omitted. `region/caption` and `region/ocr` always
require an explicit `bbox_list` plus `target_region_indexes`, since those
indexes only make sense against boxes the caller already knows.

Response shape (`InferenceResponse`):

```json
{
  "answer": "<ground>orange</ground><objects><obj0></objects>",
  "result_bbox_list": [
    {"object_index": "<obj0>", "xmin": 12.0, "ymin": 20.0, "xmax": 180.0, "ymax": 210.0, "label": "orange"}
  ],
  "prompt_tokens": 1234,
  "completion_tokens": 16,
  "bbox_list": [[12, 20, 180, 210]],
  "visualization_base64": null
}
```

## Setup

### 1. Clone with the submodule

```bash
git clone --recurse-submodules <this-repo-url>
# or, if already cloned:
git submodule update --init --recursive
```

### 2. Install dependencies

VLX-Seek's own model/runtime stack (torch, transformers, flash-attn, etc.)
is pinned in the submodule and is **not** duplicated in this repo's
`requirements.txt`, so you can match it to your hardware:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r third_party/VLX-Seek/requirements.txt   # or requirements_mac.txt on Apple Silicon
pip install -r requirements.txt                         # this service's FastAPI layer
```

### 3. Configure

```bash
cp .env.example .env
# edit .env — at minimum review VLX_API_MODEL_PATH and VLX_API_DEVICE
```

By default `VLX_API_MODEL_PATH=omlab/VLX-Seek-1.5-10B` downloads the
checkpoint from Hugging Face on first load. To use a local checkpoint
instead, point it at a directory laid out as in VLX-Seek's README
(`resources/VLX-Seek-1.5-10B/`).

Device selection follows VLX-Seek's own convention: set `VLX_API_DEVICE`
(`cuda:0`, `npu:0`, `mps`, `cpu`) or leave it unset to use `VLX_SEEK_BACKEND`
/ auto-detection (CUDA → Ascend NPU → Apple MPS → CPU).

### 4. Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The model loads lazily on the first request by default (set
`VLX_API_LOAD_MODEL_ON_STARTUP=true` to load it eagerly instead, e.g. so a
container's readiness probe only turns healthy once the model is resident).
Interactive API docs: `http://localhost:8000/docs`.

### Example request

```bash
curl -X POST http://localhost:8000/v1/detect \
  -F "image=@third_party/VLX-Seek/demo/demo_image.jpg" \
  -F "text=orange; apple" \
  -F "visualize=true"
```

## Docker

```bash
cp .env.example .env
docker compose up --build
```

`docker/Dockerfile` targets NVIDIA x86 GPUs. For Ascend NPU, Jetson, or
Apple Silicon, adapt it from the matching Dockerfile under
`third_party/VLX-Seek/docker/` (Apple Silicon MPS specifically requires
running directly on the macOS host — Docker cannot reach the host's MPS
driver).

`.dockerignore` excludes `.git/`, `.venv/`, `.env`, and `resources/` from
the build context — without it, `COPY . .` would bake local git credentials
and any downloaded weights straight into the image. Keep it in sync if you
add other local-only directories.

## Operational notes

- **Concurrency:** inference runs on a small thread pool
  (`VLX_API_MAX_CONCURRENT_INFERENCE`, default `1`) sized for a single
  GPU-resident model instance. Raise it only if you have multiple GPUs/replicas
  behind this process.
- **Auth:** set `VLX_API_API_KEY` to require `X-API-Key: <value>` on every
  `/v1/*` route. `/healthz` and `/readyz` stay open for orchestrator probes.
- **Images:** accepted as a file upload or base64 string only — there is
  deliberately no fetch-by-URL option, to avoid giving this externally
  facing API an SSRF vector. Max size is `VLX_API_MAX_IMAGE_BYTES`, enforced
  via chunked reads so an oversized payload is rejected without being fully
  buffered. `VLX_API_MAX_REQUEST_BYTES` adds a second check at the ASGI
  layer, via the `Content-Length` header, before the body is read at all
  (requests sent without a `Content-Length`, e.g. chunked transfer encoding,
  still fall through to the per-field check).
- **Generation limits:** `max_new_tokens` is capped at
  `VLX_API_MAX_NEW_TOKENS_LIMIT` (default `4096`) on every generation
  endpoint, and `bbox_list`/`target_region_indexes` are capped at
  `VLX_API_MAX_BOXES_PER_REQUEST` (default `200`) items — both guard against
  a single request tying up the (by default, single) inference worker for
  an unbounded amount of time/memory.
- **Hardware:** VLX-Seek-1.5-10B needs a GPU with substantial VRAM (BF16);
  CPU/MPS inference is supported by VLX-Seek but is far slower and mainly
  useful for development without a GPU.
- **WeDetect proposals:** the bundled detector is reloaded from its
  checkpoint on each auto-propose call (matching upstream's CLI behavior).
  For high request volume, consider caching a loaded detector instance.

## Tests

API-layer tests stub `ModelManager` with a fake worker, so they run without
GPU access or downloaded weights — they verify routing, image decoding,
validation, auth, and response shaping, not model output quality:

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

## License

This repo's FastAPI wrapper has no license file of its own yet; VLX-Seek
itself is licensed under the terms in
[third_party/VLX-Seek/LICENSE](third_party/VLX-Seek/LICENSE).
