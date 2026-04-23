"""
Crop Disease XAI Detection API
FastAPI entry point - production grade
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from models.model_loader import ModelRegistry
from routes.predict import router as predict_router

FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("crop_xai")


# ── Lifespan (startup / shutdown) ────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🌱 Starting Crop Disease XAI API …")
    registry = ModelRegistry()
    registry.load_all()                      # warm-up: load all 3 models once
    app.state.model_registry = registry
    logger.info("✅ All models loaded and cached.")
    yield
    logger.info("🛑 Shutting down …")


# ── App factory ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Crop Disease XAI API",
    description=(
        "Explainable AI-powered crop disease detection using ResNet-18, "
        "EfficientNet-B0, and DenseNet-121 with Grad-CAM, LIME, and "
        "Integrated Gradients."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
# On Render the frontend is same-origin so CORS_ORIGINS can be empty.
# For local dev, set to your frontend dev server URL.
_DEFAULT_ORIGINS = "http://localhost:3000,http://localhost:5173,http://localhost:3001"
_raw = os.getenv("CORS_ORIGINS", _DEFAULT_ORIGINS)
ALLOWED_ORIGINS: List[str] = [o.strip() for o in _raw.split(",") if o.strip()]
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = ["*"]

# ── Middleware ───────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
    expose_headers=["X-Process-Time-Ms"],
    max_age=600,
)
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.1f}"
    return response


# ── Routes ───────────────────────────────────────────────────────────────────
app.include_router(predict_router, prefix="/api/v1", tags=["Prediction"])


@app.get("/health", tags=["Health"])
async def health(request: Request):
    registry: ModelRegistry = request.app.state.model_registry
    return {
        "status": "healthy",
        "models_loaded": list(registry.loaded_models.keys()),
        "device": registry.device_name,
    }


# ── Static frontend (must come after all API routes) ─────────────────────────
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        candidate = FRONTEND_DIST / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
else:
    @app.get("/", tags=["Health"])
    async def root():
        return {"status": "ok", "service": "Crop Disease XAI API", "version": "1.0.0"}


# ── Global exception handler ─────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred.", "error": str(exc)},
    )
