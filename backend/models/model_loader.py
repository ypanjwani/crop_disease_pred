"""
models/model_loader.py
Thread-safe ModelRegistry — loads, caches, and serves all three CNN models.
Supports optional saved weights from disk; falls back to ImageNet pretrained
weights with a re-initialised head when no checkpoint is present.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.architectures import (
    ALL_ARCHITECTURES,
    ArchitectureConfig,
    CLASS_NAMES,
    NUM_CLASSES,
)

logger = logging.getLogger(__name__)

# Weights directory — override via environment variable WEIGHTS_DIR
WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", "weights"))


class ModelRegistry:
    """
    Singleton-style registry.

    Usage
    -----
    registry = ModelRegistry()
    registry.load_all()
    model, config, device = registry.get("efficientnet_b0")
    """

    _instance: Optional["ModelRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = str(self.device)
        self.loaded_models: Dict[str, Tuple[nn.Module, ArchitectureConfig]] = {}
        logger.info(f"ModelRegistry initialised on device: {self.device}")

    # ── Public API ────────────────────────────────────────────────────────────

    def load_all(self):
        """Load (or reload) every registered architecture."""
        for key in ALL_ARCHITECTURES:
            self._load(key)

    def get(self, key: str) -> Tuple[nn.Module, ArchitectureConfig]:
        """Return a (model, config) tuple, loading on first access if needed."""
        if key not in self.loaded_models:
            self._load(key)
        return self.loaded_models[key]

    def get_all(self) -> Dict[str, Tuple[nn.Module, ArchitectureConfig]]:
        return dict(self.loaded_models)

    @property
    def class_names(self):
        return CLASS_NAMES

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load(self, key: str):
        if key not in ALL_ARCHITECTURES:
            raise ValueError(f"Unknown model key: '{key}'. Valid: {list(ALL_ARCHITECTURES)}")

        builder = ALL_ARCHITECTURES[key]
        config: ArchitectureConfig = builder()
        model = config.model

        weight_path = WEIGHTS_DIR / config.weight_key
        if weight_path.exists():
            logger.info(f"Loading weights from {weight_path}")
            state = torch.load(weight_path, map_location=self.device)
            # handle both raw state_dict and {'model_state_dict': ...} checkpoints
            state_dict = state.get("model_state_dict", state)
            model.load_state_dict(state_dict, strict=False)
        else:
            logger.warning(
                f"No checkpoint found at {weight_path}. "
                f"Using ImageNet-pretrained backbone with a random head. "
                f"Place your weights at that path for correct predictions."
            )

        model.to(self.device)
        model.eval()

        self.loaded_models[key] = (model, config)
        logger.info(f"✓ {config.name} ready on {self.device}")
