"""
scripts/evaluate_aopc.py
─────────────────────────────────────────────────────────────────────────────
Reproduces the AOPC (Area Over the Perturbation Curve) analysis from the paper.
Evaluates all three models × three XAI methods and prints a results table.

Usage
─────
python scripts/evaluate_aopc.py \
    --data_dir   /path/to/plant_disease_dataset/test \
    --weights_dir backend/weights \
    --n_samples  100 \
    --n_steps    10
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from models.architectures import ALL_ARCHITECTURES
from models.model_loader import ModelRegistry, WEIGHTS_DIR
from utils.image_utils import INFERENCE_TRANSFORM, pil_to_numpy_rgb, pil_to_tensor
from xai.gradcam import generate_grad_cam
from xai.integrated_gradients import generate_integrated_gradients
from xai.lime_explainer import generate_lime

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("aopc_eval")


# ── AOPC core ─────────────────────────────────────────────────────────────────

def compute_aopc_for_attribution(
    model: nn.Module,
    input_tensor: torch.Tensor,   # (1, 3, 224, 224)
    attribution: np.ndarray,       # (H, W) — importance map
    target_idx: int,
    device: torch.device,
    n_steps: int = 10,
) -> float:
    """
    AOPC = (1 / (L+1)) * Σ_{k=0}^{L} [f(x) - f(x^(k))]
    where x^(k) is the input with the top-k most important pixels zeroed out.
    """
    model.eval()
    x = input_tensor.clone().to(device)

    with torch.no_grad():
        baseline_prob = torch.softmax(model(x), dim=1)[0, target_idx].item()

    # Flatten and sort attribution pixels by descending importance
    flat_attr = attribution.flatten()
    sorted_idx = np.argsort(flat_attr)[::-1]   # descending

    n_pixels    = len(flat_attr)
    step_size   = max(1, n_pixels // n_steps)

    drops = [0.0]   # k=0: no masking, drop = 0 by definition

    x_perturbed = x.clone()
    for k in range(1, n_steps + 1):
        # Zero-out the next batch of top-k pixels
        pixel_indices = sorted_idx[(k - 1) * step_size : k * step_size]
        rows = pixel_indices // 224
        cols = pixel_indices %  224
        x_perturbed[0, :, rows, cols] = 0.0

        with torch.no_grad():
            perturbed_prob = torch.softmax(model(x_perturbed), dim=1)[0, target_idx].item()

        drops.append(baseline_prob - perturbed_prob)

    return float(np.mean(drops))


def attribution_to_heatmap(model, pil_img, target_idx, device, method):
    """Return a (224, 224) float attribution heatmap for the given XAI method."""
    import importlib, io, base64
    import cv2

    # Reuse the XAI modules but extract the raw array instead of the overlay image
    input_tensor = pil_to_tensor(pil_img).to(device)

    if method == "gradcam":
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        config = [c for k, c in [
            (k, ALL_ARCHITECTURES[k]()) for k in ALL_ARCHITECTURES
        ] if c.model.__class__.__name__ == model.__class__.__name__][0]
        # Use the registered target layers
        layers = config.grad_cam_layers
        # Rebuild target layers pointing to actual model
        layers = _get_grad_cam_layers(model)
        with GradCAM(model=model, target_layers=layers) as cam:
            heatmap = cam(input_tensor, targets=[ClassifierOutputTarget(target_idx)])[0]
        return heatmap   # (224, 224)

    elif method == "ig":
        from captum.attr import IntegratedGradients
        baseline = torch.zeros_like(input_tensor)
        ig = IntegratedGradients(model)
        attr, _ = ig.attribute(
            input_tensor.requires_grad_(True),
            baselines=baseline,
            target=target_idx,
            n_steps=30,
            return_convergence_delta=True,
            internal_batch_size=4,
        )
        arr = attr.squeeze(0).detach().cpu().numpy()   # (3, 224, 224)
        heatmap = np.mean(np.abs(arr), axis=0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap

    elif method == "lime":
        from lime import lime_image
        from xai.lime_explainer import _make_predict_fn
        img_np  = np.array(pil_img.resize((224, 224))).astype(np.uint8)
        pred_fn = _make_predict_fn(model, device)
        explainer = lime_image.LimeImageExplainer(verbose=False)
        expl = explainer.explain_instance(img_np, pred_fn, top_labels=1,
                                          hide_color=0, num_samples=200, random_seed=42)
        _, mask = expl.get_image_and_mask(target_idx, positive_only=True,
                                          num_features=5, hide_rest=False)
        return mask.astype(np.float32)

    raise ValueError(f"Unknown method: {method}")


def _get_grad_cam_layers(model):
    """Return the correct target layer(s) for Grad-CAM based on model class."""
    name = model.__class__.__name__
    if "ResNet" in name:
        return [model.layer4[-1]]
    elif "EfficientNet" in name:
        return [model.features[-1]]
    elif "DenseNet" in name:
        return [model.features[-1]]
    raise ValueError(f"Unknown model class: {name}")


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate_aopc(args):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    registry = ModelRegistry()
    registry.load_all()

    test_ds = datasets.ImageFolder(args.data_dir, transform=INFERENCE_TRANSFORM)
    # Subsample for speed
    indices = np.random.choice(len(test_ds), min(args.n_samples, len(test_ds)), replace=False)
    subset  = Subset(test_ds, indices)
    loader  = DataLoader(subset, batch_size=1, shuffle=False)

    methods = ["gradcam", "ig", "lime"]
    results: Dict[str, Dict[str, list]] = {
        k: {m: [] for m in methods}
        for k in registry.loaded_models
    }

    for sample_idx, (img_tensor, label_tensor) in enumerate(loader):
        label = label_tensor.item()
        pil_img = Image.fromarray(
            (img_tensor.squeeze(0).permute(1,2,0).numpy() * 255).astype(np.uint8)
        ).resize((224, 224))

        log.info(f"Sample {sample_idx+1}/{len(subset)}")

        for model_key, (model, config) in registry.loaded_models.items():
            model.eval()
            with torch.no_grad():
                logits = model(img_tensor.to(device))
                pred_idx = logits.argmax(1).item()

            for method in methods:
                try:
                    heatmap = attribution_to_heatmap(model, pil_img, pred_idx, device, method)
                    score   = compute_aopc_for_attribution(
                        model, img_tensor, heatmap, pred_idx, device, n_steps=args.n_steps
                    )
                    results[model_key][method].append(score)
                except Exception as e:
                    log.warning(f"  {model_key}/{method} failed: {e}")

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "="*65)
    print("AOPC RESULTS — Area Over the Perturbation Curve")
    print("="*65)
    print(f"{'Model':<22}  {'Integ. Grads':>14}  {'Grad-CAM':>10}  {'LIME':>8}")
    print("-"*65)

    for model_key in results:
        _, config = registry.loaded_models[model_key]
        ig_mean   = np.mean(results[model_key]["ig"])   if results[model_key]["ig"]      else float("nan")
        gc_mean   = np.mean(results[model_key]["gradcam"]) if results[model_key]["gradcam"] else float("nan")
        lm_mean   = np.mean(results[model_key]["lime"]) if results[model_key]["lime"]    else float("nan")
        marker    = " ★" if model_key == "efficientnet_b0" else ""
        print(f"{config.name:<22}{marker}  {ig_mean:>+14.3f}  {gc_mean:>+10.3f}  {lm_mean:>+8.3f}")

    print("="*65)
    print("Positive AOPC = model focuses on disease lesions (faithful XAI)")
    print("Paper reference: EfficientNet-B0 → IG:+0.136, GC:+0.054, LIME:+0.087")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    type=Path, required=True)
    p.add_argument("--weights_dir", type=Path, default=Path("backend/weights"))
    p.add_argument("--n_samples",   type=int,  default=100)
    p.add_argument("--n_steps",     type=int,  default=10)
    args = p.parse_args()

    import os
    os.environ["WEIGHTS_DIR"] = str(args.weights_dir)
    evaluate_aopc(args)


if __name__ == "__main__":
    main()
