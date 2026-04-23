"""
routes/inference_pipeline.py
Orchestrates inference + XAI generation for a single image across all models.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import torch
import torch.nn as nn
from PIL import Image

from models.architectures import ArchitectureConfig, CLASS_NAMES
from utils.image_utils import pil_to_tensor
from utils.response_models import ModelResult, XAIVisuals
from xai.gradcam import generate_grad_cam
from xai.integrated_gradients import generate_integrated_gradients
from xai.lime_explainer import generate_lime

logger = logging.getLogger(__name__)

# EfficientNet is the recommended model per AOPC analysis in the paper
RECOMMENDED_KEY = "efficientnet_b0"


def _run_inference(
    model: nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
):
    """Return (predicted_idx, confidence, top5) for a single model."""
    model.eval()
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]   # (num_classes,)

    confidence, pred_idx = probs.max(0)

    # Top-5
    top5_vals, top5_idxs = probs.topk(5)
    top5 = [
        {"class": CLASS_NAMES[i], "confidence": float(v)}
        for i, v in zip(top5_idxs.tolist(), top5_vals.tolist())
    ]
    return int(pred_idx), float(confidence), top5


def _process_single_model(
    key: str,
    model: nn.Module,
    config: ArchitectureConfig,
    pil_image: Image.Image,
    device: torch.device,
) -> ModelResult:
    """Run inference + all XAI methods for one architecture."""
    t0 = time.perf_counter()

    input_tensor = pil_to_tensor(pil_image)
    pred_idx, confidence, top5 = _run_inference(model, input_tensor, device)
    predicted_class = CLASS_NAMES[pred_idx]

    inference_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"{config.name} → {predicted_class} ({confidence:.3f}) [{inference_ms:.1f} ms]")

    # ── XAI ──────────────────────────────────────────────────────────────────
    try:
        grad_cam_b64 = generate_grad_cam(
            model=model,
            pil_image=pil_image,
            target_layers=config.grad_cam_layers,
            target_class_idx=pred_idx,
            device=device,
        )
    except Exception as e:
        logger.error(f"Grad-CAM failed for {config.name}: {e}")
        grad_cam_b64 = ""

    try:
        lime_b64 = generate_lime(
            model=model,
            pil_image=pil_image,
            target_class_idx=pred_idx,
            device=device,
        )
    except Exception as e:
        logger.error(f"LIME failed for {config.name}: {e}")
        lime_b64 = ""

    try:
        ig_b64 = generate_integrated_gradients(
            model=model,
            pil_image=pil_image,
            target_class_idx=pred_idx,
            device=device,
        )
    except Exception as e:
        logger.error(f"Integrated Gradients failed for {config.name}: {e}")
        ig_b64 = ""

    total_ms = (time.perf_counter() - t0) * 1000

    return ModelResult(
        model_key=key,
        model_name=config.name,
        prediction=predicted_class,
        confidence=confidence,
        top5=top5,
        xai=XAIVisuals(grad_cam=grad_cam_b64, lime=lime_b64, ig=ig_b64),
        reliable=(key == RECOMMENDED_KEY),
        inference_ms=total_ms,
    )


def run_pipeline(
    loaded_models: dict,
    pil_image: Image.Image,
    device: torch.device,
    selected_keys: Optional[List[str]] = None,
) -> List[ModelResult]:
    """
    Run inference + XAI for all (or selected) models.

    Parameters
    ----------
    loaded_models  : {key: (model, config)} dict from ModelRegistry
    pil_image      : preprocessed PIL image
    device         : torch.device
    selected_keys  : optional subset of model keys to run

    Returns
    -------
    List[ModelResult] sorted so EfficientNet appears first
    """
    keys_to_run = selected_keys or list(loaded_models.keys())
    results = []

    # Run sequentially to avoid GPU memory contention on single-GPU boxes.
    # Switch to ThreadPoolExecutor if you have multiple GPUs or CPU-only.
    for key in keys_to_run:
        if key not in loaded_models:
            logger.warning(f"Model key '{key}' not found in registry — skipping.")
            continue
        model, config = loaded_models[key]
        result = _process_single_model(key, model, config, pil_image, device)
        results.append(result)

    # Sort: recommended model first, then alphabetical
    results.sort(key=lambda r: (0 if r.model_key == RECOMMENDED_KEY else 1, r.model_name))
    return results
