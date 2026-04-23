# 🌿 CropXAI — Explainable Plant Disease Detection

A production-grade full-stack web application that runs **ResNet-18**, **EfficientNet-B0**, and **DenseNet-121** simultaneously on plant leaf images, generating three XAI explanations per model and quantifying their faithfulness via the **AOPC** metric.

---

## 📁 Project Structure

```
crop-disease-xai/
├── backend/
│   ├── main.py                          ← FastAPI app entry point
│   ├── requirements.txt
│   ├── .env.example
│   ├── weights/                         ← 📂 Place your .pth files here
│   │   ├── resnet18_plant_disease.pth
│   │   ├── efficientnet_b0_plant_disease.pth
│   │   └── densenet121_plant_disease.pth
│   ├── models/
│   │   ├── __init__.py
│   │   ├── architectures.py             ← Model definitions + Grad-CAM layer targets
│   │   └── model_loader.py              ← Thread-safe singleton ModelRegistry
│   ├── xai/
│   │   ├── __init__.py
│   │   ├── gradcam.py                   ← pytorch-grad-cam wrapper
│   │   ├── lime_explainer.py            ← LIME superpixel explainer
│   │   └── integrated_gradients.py      ← Captum IG attribution
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── predict.py                   ← POST /api/v1/predict endpoint
│   │   └── inference_pipeline.py        ← Orchestration: inference → XAI
│   └── utils/
│       ├── __init__.py
│       ├── image_utils.py               ← PIL/NumPy/OpenCV helpers + base64 encoder
│       └── response_models.py           ← Pydantic API contract schemas
│
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    ├── postcss.config.js
    ├── .env.example
    └── src/
        ├── main.jsx                     ← React entry point
        ├── App.jsx                      ← Root layout + state
        ├── index.css                    ← Tailwind + global styles
        ├── components/
        │   ├── ImageUpload.jsx          ← Drag-and-drop uploader
        │   ├── ModelSelector.jsx        ← Model toggle with AOPC scores
        │   ├── ProgressIndicator.jsx    ← Animated progress with stage labels
        │   ├── ResultsDashboard.jsx     ← Main results container
        │   ├── ModelResultCard.jsx      ← Per-model card with top-5 + XAI
        │   ├── XAIPanel.jsx             ← Grad-CAM / LIME / IG side-by-side
        │   ├── ConfidenceBar.jsx        ← Animated confidence bars
        │   └── AOPCChart.jsx            ← Research-grade AOPC bar chart
        ├── services/
        │   └── api.js                   ← Axios API layer with interceptors
        └── hooks/
            └── usePrediction.js         ← Async prediction state machine
```

---

## 🚀 Quick Start

### 1 — Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy env
cp .env.example .env

# Place your trained weights
mkdir -p weights
# → copy resnet18_plant_disease.pth
# → copy efficientnet_b0_plant_disease.pth
# → copy densenet121_plant_disease.pth
# into the weights/ folder

# Start server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

> **No weights yet?** The server still starts — it uses ImageNet-pretrained backbones
> with a randomly initialised classification head. Predictions will be wrong but
> XAI visualisations will work for testing the pipeline.

### 2 — Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Copy env
cp .env.example .env

# Start dev server (proxies /api → localhost:8000)
npm run dev
```

Open **http://localhost:3000** in your browser.

---

## 🧠 Weights File Naming

| Architecture    | Expected filename                        |
|-----------------|------------------------------------------|
| ResNet-18       | `weights/resnet18_plant_disease.pth`     |
| EfficientNet-B0 | `weights/efficientnet_b0_plant_disease.pth` |
| DenseNet-121    | `weights/densenet121_plant_disease.pth`  |

Checkpoints can be either a raw `state_dict` or a dict with key `"model_state_dict"`.

Override the weights directory:
```bash
WEIGHTS_DIR=/path/to/weights uvicorn main:app ...
```

---

## 🔌 API Contract

### `POST /api/v1/predict`

**Request** — `multipart/form-data`

| Field    | Type     | Required | Description                                            |
|----------|----------|----------|--------------------------------------------------------|
| `file`   | File     | ✅       | Plant leaf image (JPEG / PNG / WebP, max 10 MB)        |
| `models` | string   | ❌       | Comma-separated keys: `resnet18,efficientnet_b0,densenet121` |

**Response** — `application/json`

```json
{
  "status": "success",
  "original_image": "data:image/png;base64,<b64>",
  "recommended_model": "efficientnet_b0",
  "class_names": ["Apple___Apple_scab", "..."],
  "model_results": [
    {
      "model_key": "efficientnet_b0",
      "model_name": "EfficientNet-B0",
      "prediction": "Corn___Northern_Leaf_Blight",
      "confidence": 0.973,
      "top5": [
        { "class": "Corn___Northern_Leaf_Blight", "confidence": 0.973 },
        { "class": "Corn___Common_rust", "confidence": 0.018 },
        "..."
      ],
      "xai": {
        "grad_cam": "data:image/png;base64,<b64>",
        "lime":     "data:image/png;base64,<b64>",
        "ig":       "data:image/png;base64,<b64>"
      },
      "reliable": true,
      "inference_ms": 842.3
    }
  ]
}
```

### Other Endpoints

| Method | Path               | Description                     |
|--------|--------------------|---------------------------------|
| GET    | `/health`          | Server health + loaded models   |
| GET    | `/api/v1/models`   | List available model keys       |
| GET    | `/api/v1/classes`  | List all 15 disease classes     |

---

## 🎨 15 Disease Classes

```
Apple___Apple_scab          Apple___Black_rot
Apple___Cedar_apple_rust    Apple___healthy
Corn___Cercospora_leaf_spot Corn___Common_rust
Corn___Northern_Leaf_Blight Corn___healthy
Grape___Black_rot           Grape___Esca_(Black_Measles)
Grape___Leaf_blight_...     Grape___healthy
Potato___Early_blight       Potato___Late_blight
Potato___healthy
```

Update `CLASS_NAMES` in `backend/models/architectures.py` to match your dataset.

---

## 📊 AOPC Research Results (from paper)

| Model           | Integrated Gradients | Grad-CAM | LIME   |
|-----------------|----------------------|----------|--------|
| ResNet-18       | -0.107               | -0.157   | -0.013 |
| **EfficientNet-B0** | **+0.136**       | **+0.054** | **+0.087** |
| DenseNet-121    | +0.067               | -0.197   | -0.128 |

✅ **EfficientNet-B0** is the only model with positive AOPC across all three XAI methods, confirming it reliably focuses on disease lesions rather than background noise.

---

## ⚙️ GPU Support

PyTorch automatically uses CUDA if available. To install the CUDA-enabled build:

```bash
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

---

## 🏗️ Production Deployment

```bash
# Backend
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2

# Frontend
npm run build
# Serve dist/ with nginx or any static host
```

For multi-worker deployments set `workers=1` (model registry is a singleton per process)
or externalise model loading to a shared GPU service.
