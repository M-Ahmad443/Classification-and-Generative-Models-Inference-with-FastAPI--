import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))

# ---------------- classification models ----------------
MODEL_REGISTRY = {
    "efficientnet_b0": {
        "file": "efficientnet_b0_garbage.pth",
        "arch": "efficientnet_b0",
        "head": "single",   # in -> 256 -> classes
        "act": "silu",
    },
    "densenet121": {
        "file": "densenet121_garbage.pth",
        "arch": "densenet121",
        "head": "double",   # in -> 256 -> 128 -> classes
        "act": "relu",
    },
}

ENABLED_MODELS = [
    m.strip()
    for m in os.getenv("ENABLED_MODELS", ",".join(MODEL_REGISTRY)).split(",")
    if m.strip()
]

# Classifiers were trained on 128x128 inputs. A checkpoint's own "img_size" wins if present.
IMG_SIZE = int(os.getenv("IMG_SIZE", "128"))

# ---------------- generator (denoising) model ----------------
# ---------------- generator (denoising) model ----------------
ENABLE_GENERATOR = os.getenv("ENABLE_GENERATOR", "1") == "1"
GENERATOR_PATH = MODELS_DIR / os.getenv("GENERATOR_FILE", "generator_best_model.pth")
GENERATOR_IMG_SIZE = int(os.getenv("GENERATOR_IMG_SIZE", "224"))
GENERATOR_NORM = os.getenv("GENERATOR_NORM", "none")
GENERATOR_OUTPUT = os.getenv("GENERATOR_OUTPUT", "01")
GENERATOR_NOTE = (
    "Generalized denoising model: handles salt-and-pepper, Poisson, Gaussian and "
    "speckle noise, and mixed combinations of them, at a moderate noise level."
)

# ---------------- upload limits ----------------
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "5"))
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}
MIN_IMAGE_SIDE = 32      # reject tiny/garbage images
MAX_IMAGE_SIDE = 4096    # reject huge images (memory protection)