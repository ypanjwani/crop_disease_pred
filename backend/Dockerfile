# ── Stage 1: Build React frontend ─────────────────────────────────────────────
FROM node:20-slim AS frontend-builder

WORKDIR /frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


# ── Stage 2: Install Python deps ──────────────────────────────────────────────
FROM python:3.11-slim AS python-builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 3: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=python-builder /install /usr/local

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy Python source
COPY . .

# Copy built frontend into expected location
COPY --from=frontend-builder /frontend/dist ./frontend/dist

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

VOLUME ["/app/weights"]

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WEIGHTS_DIR=/app/weights \
    PORT=8000

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
