"""
utils/image_utils.py
Image preprocessing, tensor conversion, and base64 encoding helpers.
"""

import base64
import io
from typing import Tuple, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# ── ImageNet normalization constants ─────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
INPUT_SIZE    = 224

# ── Standard inference transform ─────────────────────────────────────────────
INFERENCE_TRANSFORM = T.Compose([
    T.Resize((INPUT_SIZE, INPUT_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_pil_image(image_bytes: bytes) -> Image.Image:
    """Decode raw bytes to a PIL RGB image."""
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """Transform a PIL image → normalised 4-D tensor (1, C, H, W)."""
    return INFERENCE_TRANSFORM(pil_image).unsqueeze(0)


def pil_to_numpy_rgb(pil_image: Image.Image, size: int = INPUT_SIZE) -> np.ndarray:
    """Resize PIL image and return a (H, W, 3) float32 array in [0, 1]."""
    img = pil_image.resize((size, size))
    return np.array(img, dtype=np.float32) / 255.0


def numpy_to_pil(array: np.ndarray) -> Image.Image:
    """Convert a numpy array (H, W, 3) in [0,1] or [0,255] to PIL."""
    if array.max() <= 1.0:
        array = (array * 255).astype(np.uint8)
    return Image.fromarray(array.astype(np.uint8))


def overlay_heatmap_on_image(
    heatmap: np.ndarray,
    original_rgb: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Blend a grayscale heatmap with the original RGB image.

    Parameters
    ----------
    heatmap      : (H, W) float in [0, 1]
    original_rgb : (H, W, 3) float in [0, 1]
    alpha        : blending factor for the heatmap layer
    colormap     : OpenCV colormap constant

    Returns
    -------
    (H, W, 3) uint8 overlay image
    """
    h, w = original_rgb.shape[:2]

    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored = cv2.applyColorMap(heatmap_uint8, colormap)   # BGR
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    original_uint8 = np.uint8(original_rgb * 255)
    overlay = np.uint8(alpha * colored_rgb + (1 - alpha) * original_uint8)
    return overlay


def encode_image_to_base64(image: Union[np.ndarray, Image.Image], fmt: str = "PNG") -> str:
    """
    Encode a numpy array or PIL image to a base64 data-URI string.
    Returns: "data:image/png;base64,<b64>"
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "png" if fmt.upper() == "PNG" else "jpeg"
    return f"data:image/{mime};base64,{b64}"


def resize_to_square(pil_image: Image.Image, size: int = INPUT_SIZE) -> Image.Image:
    return pil_image.resize((size, size), Image.LANCZOS)
