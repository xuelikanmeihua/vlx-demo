# VLX-Seek inference API, NVIDIA x86 GPUs. For other hardware (Ascend NPU,
# Jetson, Apple Silicon), adapt from third_party/VLX-Seek/docker/Dockerfile.*
# and swap the base image / requirements file below.
FROM nvcr.io/nvidia/pytorch:26.08-py3

WORKDIR /workspace

# VLX-Seek's own runtime deps (torch/transformers/flash-attn/...).
COPY third_party/VLX-Seek/requirements.txt /tmp/vlx_seek_requirements.txt
RUN pip install --no-cache-dir -r /tmp/vlx_seek_requirements.txt

# This service's FastAPI layer.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . .

ENV VLX_API_HOST=0.0.0.0 \
    VLX_API_PORT=8000

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
