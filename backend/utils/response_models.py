"""
utils/response_models.py
Pydantic schemas that define the full API contract.
"""

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class TopPrediction(BaseModel):
    """One entry in the top-5 list."""
    model_config = ConfigDict(populate_by_name=True)

    class_name: str   = Field(..., alias="class", description="Class label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Softmax probability")


class XAIVisuals(BaseModel):
    """Base64-encoded explanation images for one model."""
    grad_cam: str = Field(..., description="Grad-CAM heatmap overlay as base64 data-URI")
    lime: str     = Field(..., description="LIME superpixel explanation as base64 data-URI")
    ig: str       = Field(..., description="Integrated Gradients attribution map as base64 data-URI")


class ModelResult(BaseModel):
    """Inference + XAI result for a single model."""
    model_key:   str   = Field(..., description="Internal key, e.g. 'efficientnet_b0'")
    model_name:  str   = Field(..., description="Human-readable name, e.g. 'EfficientNet-B0'")
    prediction:  str   = Field(..., description="Predicted class label")
    confidence:  float = Field(..., ge=0.0, le=1.0, description="Softmax confidence [0,1]")
    top5: List[TopPrediction] = Field(
        default_factory=list,
        description="Top-5 class predictions with confidence scores",
    )
    xai:         XAIVisuals
    reliable:    bool  = Field(
        False,
        description="True for EfficientNet-B0 — the most faithful model per AOPC analysis",
    )
    inference_ms: float = Field(..., description="Inference time in milliseconds")


class PredictionResponse(BaseModel):
    """Top-level response returned by POST /api/v1/predict."""
    status:         str              = "success"
    original_image: str              = Field(..., description="Uploaded image as base64 data-URI")
    model_results:  List[ModelResult]
    recommended_model: str           = Field(
        "efficientnet_b0",
        description="Model recommended based on AOPC fidelity research",
    )
    class_names: List[str]


class ErrorResponse(BaseModel):
    status:  str = "error"
    detail:  str
    error:   Optional[str] = None
