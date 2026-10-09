<div align="center">

# ♻️ Garbage Classification & Denoising API

**One FastAPI service. Three deep-learning models. Classify waste, restore noisy images.**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch&logoColor=white)
![Models](https://img.shields.io/badge/Models-EfficientNet--B0%20%7C%20DenseNet121%20%7C%20CNN%2BViT-blue)

</div>

---

## ✨ Overview

This project turns research notebooks into a clean, deployable REST API.
Upload an image and get back:

- **Waste classification** into 6 classes: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`,
  from two interchangeable models (**EfficientNet-B0** and **DenseNet121**).
- **Image denoising**: a hybrid **CNN + Vision Transformer** generator restores noisy images and
  returns the input and result side by side.

The denoiser is a generalized model trained to handle **salt-and-pepper, Poisson, Gaussian and speckle
noise, and mixed combinations of them, at a moderate noise level**.

## 🚀 Features

- **Two classification endpoints**: pick the model per request, same response format.
- **Full probability output**: confidence plus probabilities for all six classes.
- **Denoising with a visual result**: JSON (base64 images) or a ready-to-view side-by-side PNG.
- **Models load once at startup** and are warmed up, so the first request is not slow.
- **Fail-soft design**: if the generator fails to load, the classifiers keep working.
- **Request tracing**: every response carries an `X-Request-ID`, and every request is logged with latency.
- **Input validation**: content type, empty file, size limit, corrupt image, minimum and maximum
  resolution (protects server memory). Phone-photo rotation (EXIF) is corrected automatically.
- **Interactive docs** at `/docs` (Swagger UI): upload and test from the browser.
- **Test suite** with pytest covering success and error paths.
- **Configurable through environment variables**, with no code edits needed.

## 🧠 Models

| Model | Task | Details |
|---|---|---|
| **EfficientNet-B0** | Classification | ImageNet-pretrained backbone, head: `Linear(in,256) → SiLU → Dropout(0.4) → Linear(256,6)`, 128×128 input |
| **DenseNet121** | Classification | ImageNet-pretrained backbone, head: `Linear(in,256) → ReLU → Dropout(0.4) → Linear(256,128) → ReLU → Dropout(0.3) → Linear(128,6)`, 128×128 input |
| **Hybrid Generator** | Denoising | CNN encoder/decoder with a ViT branch, stage-wise fusion and a learned guided-fusion weight; 224×224 input |

### Classification results (test set, 2,086 images)

| Model | Accuracy | Macro F1 |
|---|---|---|
| DenseNet121 | 0.9487 | 0.9486 |
| EfficientNet-B0 | ≈ 0.94 | ≈ 0.94 |

> For comparison, a custom 4.85 M-parameter CNN trained from scratch reached 0.5523 under the same
> limited compute budget, which shows the value of transfer learning on this task.

### Generator architecture

```
Noisy image ──► CNN Encoder ──────────────┐ skip features
      │                                   ▼
      └─► ViT Extractor ─► ViT Decoder ─► CNN Decoder ──► cnn_out ─┐
                               │                                    ├─► guided fusion ─► denoised image
                               └────────────────────► vit_out ─────┘   (learned sigmoid weight)
```

## 📁 Project structure

```
garbage-api/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app, lifespan, middleware, routes
│   ├── config.py          # paths, limits, model registry, env vars
│   ├── inference.py       # GarbageClassifier, model builder, image validation
│   ├── generator_net.py   # Hybrid CNN+ViT generator architecture
│   ├── generator.py       # Generator loading, preprocessing, inference, plotting
│   └── schemas.py         # Pydantic response models
├── models/                # .pth checkpoints (see "Model files")
├── tests/
│   └── test_api.py
├── requirements.txt
├── pytest.ini
└── README.md
```

## ⚙️ Installation (Windows)

Requires **Python 3.12**.

```cmd
git clone https://github.com/YOUR-USERNAME/garbage-api.git
cd garbage-api

python -m venv venv
venv\Scripts\activate

python -m pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

On Linux or macOS, activate with `source venv/bin/activate` instead.

### Model files

Place these checkpoints in the `models/` folder:

```
models/
├── efficientnet_b0_garbage.pth
├── densenet121_garbage.pth
└── generator_best_model.pth
```

## ▶️ Run

```cmd
python -m uvicorn app.main:app --reload --port 8000
```

Wait for the three `Loaded ...` log lines, then open:

- **Interactive docs:** http://127.0.0.1:8000/docs
- **Health check:** http://127.0.0.1:8000/health

## 🔌 API reference

| Method | Route | Description |
|---|---|---|
| `GET` | `/health` | Service status and which models are loaded |
| `GET` | `/models` | Classifier names, classes and input sizes |
| `POST` | `/predict/efficientnet_b0` | Classify an image with EfficientNet-B0 |
| `POST` | `/predict/densenet121` | Classify an image with DenseNet121 |
| `GET` | `/generate/info` | Generator details and its model note |
| `POST` | `/generate/denoise` | Denoise an image; returns input and output as base64 PNG |
| `POST` | `/generate/denoise/plot` | Denoise an image; returns one side-by-side PNG |

### Classify an image

```cmd
curl -X POST "http://127.0.0.1:8000/predict/efficientnet_b0" -F "file=@test.jpg"
```

```json
{
  "request_id": "a1b2c3d4e5f6",
  "model": "efficientnet_b0",
  "predicted_class": "plastic",
  "confidence": 0.9132,
  "probabilities": {
    "cardboard": 0.0100, "glass": 0.0200, "metal": 0.0300,
    "paper": 0.0100, "plastic": 0.9132, "trash": 0.0168
  },
  "img_size": 128,
  "latency_ms": 18.4
}
```

### Denoise an image

Save the side-by-side comparison straight to a file:

```cmd
curl -X POST "http://127.0.0.1:8000/generate/denoise/plot" -F "file=@noisy.png" -o comparison.png
```

Or use the JSON route (`/generate/denoise`) to get both images as base64 strings together with the
model note, latency and request ID.

### Error codes

| Code | Meaning |
|---|---|
| `400` | Empty, corrupt, too small (< 32 px) or too large (> 4096 px) image |
| `413` | File larger than the upload limit (default 5 MB) |
| `415` | Unsupported file type (allowed: JPEG, PNG, BMP, WebP) |
| `503` | Requested model is not loaded |

## 🔧 Configuration

All settings are environment variables. In Windows `cmd`, set them before starting the server,
for example `set ENABLED_MODELS=efficientnet_b0`.

| Variable | Default | Purpose |
|---|---|---|
| `MODELS_DIR` | `models/` | Folder containing the checkpoints |
| `ENABLED_MODELS` | `efficientnet_b0,densenet121` | Classifiers to load |
| `IMG_SIZE` | `128` | Fallback classifier input size |
| `ENABLE_GENERATOR` | `1` | Set to `0` to disable the denoiser |
| `GENERATOR_FILE` | `generator_best_model.pth` | Generator checkpoint name |
| `GENERATOR_IMG_SIZE` | `224` | Generator input size (the ViT branch requires 224) |
| `GENERATOR_NORM` | `none` | Input normalization used in training (`none`, `half`, `imagenet`) |
| `GENERATOR_OUTPUT` | `01` | Output range of the generator (`01` or `tanh`) |
| `MAX_UPLOAD_MB` | `5` | Maximum upload size |

## 🧪 Tests

```cmd
python -m pytest -v
```

The tests cover valid predictions for both classifiers and the denoiser, plus wrong file types, empty
files, corrupt images, tiny images and oversized uploads. Tests that need a model are skipped
automatically when its checkpoint is missing.

## ⚠️ Limitations

- Classifiers resize every image to 128×128 and the denoiser to 224×224, so non-square images
  are stretched.
- The classifier knows only the six trained classes, and it always returns one of them, even for
  images that are not waste.
- The denoiser targets moderate noise. Very heavy noise or noise types outside the four listed
  may give weaker results.
- Class balance, image style and background in the training data affect real-world accuracy.

## 🗺️ Roadmap

- [ ] Dockerfile and one-command deployment
- [ ] `/predict/compare` endpoint to run both classifiers on one image
- [ ] Optional synthetic noise (Gaussian, salt-and-pepper, speckle, Poisson, or all) for clean uploads
- [ ] Batch prediction
- [ ] Confusion-matrix and metrics report in the repo

## 🛠️ Tech stack

FastAPI · Uvicorn · PyTorch · torchvision · timm · Pillow · Pydantic · pytest

## 👤 Author

**YOUR NAME**: [GitHub](https://github.com/YOUR-USERNAME) · [LinkedIn](https://linkedin.com/in/YOUR-PROFILE)

## 📄 License

Add a license file (for example MIT) and state it here.