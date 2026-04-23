"""
models/architectures.py
Defines all three CNN architectures with their XAI-compatible layer targets.
"""

from dataclasses import dataclass
from typing import Any, List

import torch
import torch.nn as nn
import torchvision.models as tv_models


NUM_CLASSES = 15

CLASS_NAMES: List[str] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn___Cercospora_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
]


@dataclass
class ArchitectureConfig:
    name: str                         # human-readable key
    model: nn.Module                  # PyTorch model
    grad_cam_layers: List[Any]        # target layers for Grad-CAM
    weight_key: str                   # filename of saved weights


def build_resnet18() -> ArchitectureConfig:
    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return ArchitectureConfig(
        name="ResNet-18",
        model=model,
        grad_cam_layers=[model.layer4[-1]],
        weight_key="resnet18_plant_disease.pth",
    )


def build_efficientnet_b0() -> ArchitectureConfig:
    model = tv_models.efficientnet_b0(
        weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    return ArchitectureConfig(
        name="EfficientNet-B0",
        model=model,
        grad_cam_layers=[model.features[-1]],
        weight_key="efficientnet_b0_plant_disease.pth",
    )


def build_densenet121() -> ArchitectureConfig:
    model = tv_models.densenet121(
        weights=tv_models.DenseNet121_Weights.IMAGENET1K_V1
    )
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, NUM_CLASSES)
    return ArchitectureConfig(
        name="DenseNet-121",
        model=model,
        grad_cam_layers=[model.features[-1]],
        weight_key="densenet121_plant_disease.pth",
    )


ALL_ARCHITECTURES = {
    "resnet18": build_resnet18,
    "efficientnet_b0": build_efficientnet_b0,
    "densenet121": build_densenet121,
}
