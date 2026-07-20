FROM python:3.11-slim

# ── System dependencies ──────────────────────────────────────────────────────
# libgl1          : OpenCV / libGL dependency
# libglib2.0-0    : OpenCV dependency (GTK backend, glib)
# wget curl       : optional tooling / healthcheck convenience
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        wget \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────────────────────
COPY app/ ./app/

# Pre-create model directory (model is downloaded at first startup if absent)
RUN mkdir -p /app/models

# ── Environment ──────────────────────────────────────────────────────────────
ENV MODEL_PATH=/app/models/yolov8n.onnx \
    ORT_THREADS=4 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── Expose & run ─────────────────────────────────────────────────────────────
EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", \
     "--workers", "1", "--log-level", "info"]
