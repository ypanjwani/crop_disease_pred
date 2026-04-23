"""
xai/gradcam.py
Grad-CAM explanation using pytorch-grad-cam library.
Supports ResNet-18, EfficientNet-B0, and DenseNet-121.
"""

import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from utils.image_utils import (
    encode_image_to_base64,
    overlay_heatmap_on_image,
    pil_to_numpy_rgb,
    pil_to_tensor,
)
from PIL import Image

logger = logging.getLogger(__name__)


def generate_grad_cam(
    model: nn.Module,
    pil_image: Image.Image,
    target_layers: List[nn.Module],
    target_class_idx: int,
    device: torch.device,
) -> str:
    """
    Generate a Grad-CAM heatmap overlay for the given model and image.

    Returns
    -------
    str  — base64 data-URI of the overlay image (PNG)
    """
    model.eval()
    input_tensor = pil_to_tensor(pil_image).to(device)           # (1, 3, 224, 224)
    original_rgb = pil_to_numpy_rgb(pil_image)                    # (224, 224, 3) float [0,1]

    targets = [ClassifierOutputTarget(target_class_idx)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets,
            aug_smooth=True,        # average over test-time augmentations
            eigen_smooth=True,      # first-principal-component smoothing
        )                           # returns (B, H, W)

    heatmap = grayscale_cam[0]      # (H, W) float in [0, 1]
    overlay = overlay_heatmap_on_image(heatmap, original_rgb, alpha=0.45)
    return encode_image_to_base64(overlay)
