"""
tests/test_api.py
─────────────────────────────────────────────────────────────────────────────
Integration tests for the CropXAI FastAPI backend.

Run with:
    cd backend
    pytest ../tests/ -v
"""

import base64
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

# Allow imports from backend/
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_test_image(size=(224, 224), color="green") -> bytes:
    """Create a synthetic leaf-like image for testing."""
    img  = Image.new("RGB", size, color=(34, 139, 34))
    draw = ImageDraw.Draw(img)
    # Draw some spots to simulate disease lesions
    for x, y in [(60, 60), (120, 100), (80, 160)]:
        draw.ellipse([x-15, y-15, x+15, y+15], fill=(180, 60, 20))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    """Create a test client with models loaded."""
    from main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def test_image_bytes():
    return _make_test_image()


# ── Health check tests ────────────────────────────────────────────────────────

class TestHealth:
    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_endpoint(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert "models_loaded" in data
        assert isinstance(data["models_loaded"], list)
        assert len(data["models_loaded"]) == 3

    def test_models_list(self, client):
        r = client.get("/api/v1/models")
        assert r.status_code == 200
        models = r.json()["models"]
        keys = [m["key"] for m in models]
        assert "resnet18" in keys
        assert "efficientnet_b0" in keys
        assert "densenet121" in keys

    def test_classes_list(self, client):
        r = client.get("/api/v1/classes")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 15
        assert len(data["classes"]) == 15


# ── Prediction endpoint tests ─────────────────────────────────────────────────

class TestPredict:
    def test_predict_all_models(self, client, test_image_bytes):
        r = client.post(
            "/api/v1/predict",
            files={"file": ("leaf.png", test_image_bytes, "image/png")},
        )
        assert r.status_code == 200
        data = r.json()

        assert data["status"] == "success"
        assert "original_image" in data
        assert data["original_image"].startswith("data:image/")
        assert len(data["model_results"]) == 3
        assert data["recommended_model"] == "efficientnet_b0"

    def test_predict_single_model(self, client, test_image_bytes):
        r = client.post(
            "/api/v1/predict",
            files={"file": ("leaf.png", test_image_bytes, "image/png")},
            data={"models": "efficientnet_b0"},
        )
        assert r.status_code == 200
        results = r.json()["model_results"]
        assert len(results) == 1
        assert results[0]["model_key"] == "efficientnet_b0"

    def test_predict_model_result_schema(self, client, test_image_bytes):
        r = client.post(
            "/api/v1/predict",
            files={"file": ("leaf.png", test_image_bytes, "image/png")},
            data={"models": "efficientnet_b0"},
        )
        result = r.json()["model_results"][0]

        # Required fields
        assert "model_key"    in result
        assert "model_name"   in result
        assert "prediction"   in result
        assert "confidence"   in result
        assert "top5"         in result
        assert "xai"          in result
        assert "reliable"     in result
        assert "inference_ms" in result

        # Confidence in [0, 1]
        assert 0.0 <= result["confidence"] <= 1.0

        # Top-5 should have 5 entries
        assert len(result["top5"]) == 5
        for item in result["top5"]:
            assert "class"      in item
            assert "confidence" in item

        # XAI images should be base64 data URIs
        xai = result["xai"]
        for key in ("grad_cam", "lime", "ig"):
            assert key in xai
            assert xai[key].startswith("data:image/")

    def test_efficientnet_marked_reliable(self, client, test_image_bytes):
        r = client.post(
            "/api/v1/predict",
            files={"file": ("leaf.png", test_image_bytes, "image/png")},
        )
        results = {res["model_key"]: res for res in r.json()["model_results"]}
        assert results["efficientnet_b0"]["reliable"] is True
        assert results["resnet18"]["reliable"]        is False
        assert results["densenet121"]["reliable"]     is False

    def test_efficientnet_first_in_response(self, client, test_image_bytes):
        r = client.post(
            "/api/v1/predict",
            files={"file": ("leaf.png", test_image_bytes, "image/png")},
        )
        first = r.json()["model_results"][0]
        assert first["model_key"] == "efficientnet_b0"

    def test_xai_images_are_valid_png(self, client, test_image_bytes):
        r = client.post(
            "/api/v1/predict",
            files={"file": ("leaf.png", test_image_bytes, "image/png")},
            data={"models": "efficientnet_b0"},
        )
        xai = r.json()["model_results"][0]["xai"]
        for key, uri in xai.items():
            header, b64 = uri.split(",", 1)
            raw = base64.b64decode(b64)
            img = Image.open(io.BytesIO(raw))
            assert img.size == (224, 224), f"{key} image is wrong size: {img.size}"


# ── Error handling tests ──────────────────────────────────────────────────────

class TestErrors:
    def test_invalid_file_type(self, client):
        r = client.post(
            "/api/v1/predict",
            files={"file": ("doc.pdf", b"%PDF-fake", "application/pdf")},
        )
        assert r.status_code == 415

    def test_invalid_model_key(self, client, test_image_bytes):
        r = client.post(
            "/api/v1/predict",
            files={"file": ("leaf.png", test_image_bytes, "image/png")},
            data={"models": "fakenet_99"},
        )
        assert r.status_code == 422

    def test_corrupted_image(self, client):
        r = client.post(
            "/api/v1/predict",
            files={"file": ("bad.png", b"\x00\x01\x02\x03garbage", "image/png")},
        )
        assert r.status_code == 400

    def test_multiple_valid_models(self, client, test_image_bytes):
        r = client.post(
            "/api/v1/predict",
            files={"file": ("leaf.png", test_image_bytes, "image/png")},
            data={"models": "resnet18,densenet121"},
        )
        assert r.status_code == 200
        keys = [res["model_key"] for res in r.json()["model_results"]]
        assert "resnet18"    in keys
        assert "densenet121" in keys
        assert "efficientnet_b0" not in keys


# ── Utility tests ─────────────────────────────────────────────────────────────

class TestImageUtils:
    def test_encode_decode_roundtrip(self):
        from utils.image_utils import encode_image_to_base64, load_pil_image

        img_bytes = _make_test_image()
        pil_img   = load_pil_image(img_bytes)
        b64       = encode_image_to_base64(pil_img)

        assert b64.startswith("data:image/png;base64,")
        _, raw_b64 = b64.split(",", 1)
        decoded = Image.open(io.BytesIO(base64.b64decode(raw_b64)))
        assert decoded.mode == "RGB"

    def test_overlay_heatmap(self):
        from utils.image_utils import overlay_heatmap_on_image

        heatmap = np.random.rand(14, 14).astype(np.float32)
        img_rgb = np.random.rand(224, 224, 3).astype(np.float32)
        result  = overlay_heatmap_on_image(heatmap, img_rgb)
        assert result.shape == (224, 224, 3)
        assert result.dtype == np.uint8
