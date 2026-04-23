"""
routes/predict.py
FastAPI router for the /predict endpoint.
"""

import logging
from functools import partial
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from models.architectures import CLASS_NAMES
from models.model_loader import ModelRegistry
from routes.inference_pipeline import run_pipeline
from utils.image_utils import encode_image_to_base64, load_pil_image
from utils.response_models import PredictionResponse

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_FILE_SIZE_BYTES   = 10 * 1024 * 1024   # 10 MB


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict plant disease + generate XAI explanations",
    description=(
        "Upload a leaf image. The API runs all three CNN models (or a selected subset), "
        "generates Grad-CAM, LIME, and Integrated Gradients explanations, and returns "
        "structured results including base64-encoded overlay images."
    ),
)
async def predict(
    request: Request,
    file: UploadFile = File(..., description="Plant leaf image (JPEG/PNG/WebP)"),
    models: Optional[str] = Form(
        default=None,
        description=(
            "Comma-separated list of model keys to run. "
            "Valid values: resnet18, efficientnet_b0, densenet121. "
            "Omit to run all three."
        ),
    ),
):
    # ── Validate upload ───────────────────────────────────────────────────────
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG, PNG, WebP, or BMP.",
        )

    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10 MB.")

    # ── Parse requested models ────────────────────────────────────────────────
    selected_keys = None
    if models:
        selected_keys = [k.strip() for k in models.split(",") if k.strip()]
        valid_keys = {"resnet18", "efficientnet_b0", "densenet121"}
        invalid = set(selected_keys) - valid_keys
        if invalid:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown model key(s): {invalid}. Valid: {valid_keys}",
            )

    # ── Load image ────────────────────────────────────────────────────────────
    try:
        pil_image = load_pil_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")

    original_b64 = encode_image_to_base64(pil_image.resize((224, 224)))

    # ── Retrieve registry (set during lifespan startup) ───────────────────────
    registry: ModelRegistry = request.app.state.model_registry

    # ── Run pipeline (offload blocking CPU work off the event loop) ───────────
    try:
        results = await run_in_threadpool(
            partial(
                run_pipeline,
                loaded_models=registry.get_all(),
                pil_image=pil_image,
                device=registry.device,
                selected_keys=selected_keys,
            )
        )
    except Exception as e:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=f"Inference pipeline failed: {e}")

    if not results:
        raise HTTPException(status_code=500, detail="No model results were produced.")

    return PredictionResponse(
        status="success",
        original_image=original_b64,
        model_results=results,
        recommended_model="efficientnet_b0",
        class_names=CLASS_NAMES,
    )


@router.get(
    "/models",
    summary="List available models",
)
async def list_models(request: Request):
    registry: ModelRegistry = request.app.state.model_registry
    return {
        "models": [
            {
                "key": key,
                "name": config.name,
                "recommended": key == "efficientnet_b0",
            }
            for key, (_, config) in registry.get_all().items()
        ]
    }


@router.get("/classes", summary="List all plant disease classes")
async def list_classes():
    return {"classes": CLASS_NAMES, "count": len(CLASS_NAMES)}
