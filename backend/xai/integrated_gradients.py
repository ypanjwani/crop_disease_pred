"""
xai/integrated_gradients.py
Integrated Gradients attribution using Captum.
Uses a black-image baseline and accumulates gradients along the interpolation path.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients

from utils.image_utils import (
    encode_image_to_base64,
    overlay_heatmap_on_image,
    pil_to_numpy_rgb,
    pil_to_tensor,
)
from PIL import Image

logger = logging.getLogger(__name__)

N_STEPS          = 50      # interpolation steps (accuracy vs speed)
IG_THRESHOLD     = 0.20    # suppress attribution noise below this quantile


def _normalise_attribution(attr: np.ndarray) -> np.ndarray:
    """
    Collapse channel dimension and normalise to [0, 1].
    attr shape: (C, H, W) or (H, W)
    """
    if attr.ndim == 3:
        attr = np.mean(np.abs(attr), axis=0)   # (H, W)

    # percentile-based normalisation — robust to outliers
    vmin = np.percentile(attr, 1)
    vmax = np.percentile(attr, 99)
    attr = np.clip((attr - vmin) / (vmax - vmin + 1e-8), 0, 1)

    # suppress noise below threshold
    attr[attr < IG_THRESHOLD] = 0.0
    return attr


def generate_integrated_gradients(
    model: nn.Module,
    pil_image: Image.Image,
    target_class_idx: int,
    device: torch.device,
) -> str:
    """
    Generate an Integrated Gradients attribution map overlay.

    Returns
    -------
    str  — base64 data-URI of the attribution overlay (PNG)
    """
    model.eval()

    input_tensor = pil_to_tensor(pil_image).to(device).requires_grad_(True)   # (1,3,224,224)
    baseline     = torch.zeros_like(input_tensor)                             # black baseline

    original_rgb = pil_to_numpy_rgb(pil_image)                                # (224,224,3)

    ig = IntegratedGradients(model)
    attributions, _ = ig.attribute(
        input_tensor,
        baselines=baseline,
        target=target_class_idx,
        n_steps=N_STEPS,
        return_convergence_delta=True,
        internal_batch_size=8,
    )                              # (1, 3, 224, 224)

    attr_np = attributions.squeeze(0).detach().cpu().numpy()   # (3, 224, 224)
    heatmap = _normalise_attribution(attr_np)                  # (224, 224) in [0, 1]

    overlay = overlay_heatmap_on_image(heatmap, original_rgb, alpha=0.55)
    return encode_image_to_base64(overlay)
