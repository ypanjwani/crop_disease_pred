"""
xai/lime_explainer.py
LIME (Local Interpretable Model-agnostic Explanations) for image classification.
Uses superpixel segmentation to produce human-interpretable explanations.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from lime import lime_image
from skimage.segmentation import mark_boundaries

from utils.image_utils import (
    INFERENCE_TRANSFORM,
    encode_image_to_base64,
    pil_to_numpy_rgb,
)
from PIL import Image

logger = logging.getLogger(__name__)

NUM_SAMPLES    = 500   # number of perturbed samples (stability vs speed trade-off)
TOP_FEATURES   = 5     # number of top superpixels to highlight
HIDE_REST      = True  # hide non-contributing pixels for cleaner visualisation


def _make_predict_fn(model: nn.Module, device: torch.device):
    """
    Build a prediction function that LIME can call with batches of numpy images.
    Input:  numpy array of shape (N, H, W, 3) uint8 / float in [0,255]
    Output: numpy array of shape (N, num_classes) — softmax probabilities
    """
    def predict(images: np.ndarray) -> np.ndarray:
        model.eval()
        batch = []
        for img in images:
            pil = Image.fromarray(img.astype(np.uint8)).convert("RGB")
            tensor = INFERENCE_TRANSFORM(pil)
            batch.append(tensor)
        batch_tensor = torch.stack(batch).to(device)         # (N, 3, 224, 224)

        with torch.no_grad():
            logits = model(batch_tensor)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    return predict


def generate_lime(
    model: nn.Module,
    pil_image: Image.Image,
    target_class_idx: int,
    device: torch.device,
) -> str:
    """
    Generate a LIME superpixel explanation overlay.

    Returns
    -------
    str  — base64 data-URI of the LIME boundary image (PNG)
    """
    model.eval()

    # LIME works in uint8 pixel space
    img_np = np.array(pil_image.resize((224, 224))).astype(np.uint8)   # (224, 224, 3)

    predict_fn = _make_predict_fn(model, device)

    explainer   = lime_image.LimeImageExplainer(verbose=False)
    explanation = explainer.explain_instance(
        img_np,
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=NUM_SAMPLES,
        batch_size=32,
        random_seed=42,
    )

    # Extract mask of top contributing superpixels
    temp_img, mask = explanation.get_image_and_mask(
        label=target_class_idx,
        positive_only=True,
        num_features=TOP_FEATURES,
        hide_rest=HIDE_REST,
    )

    # Draw green boundaries around contributing superpixels
    overlay = mark_boundaries(temp_img / 255.0, mask, color=(0, 1, 0), mode="thick")
    overlay_uint8 = (overlay * 255).astype(np.uint8)
    return encode_image_to_base64(overlay_uint8)
