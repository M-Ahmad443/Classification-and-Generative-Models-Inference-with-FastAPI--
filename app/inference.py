import io
import logging

import timm
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms

logger = logging.getLogger("garbage-api")

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _act(name: str) -> nn.Module:
    return nn.SiLU() if name == "silu" else nn.ReLU()


def build_model(arch: str, num_classes: int, head: str = "single", act: str = "silu") -> nn.Module:
    """Must match the training-time architecture exactly."""
    model = timm.create_model(arch, pretrained=False)
    in_features = model.classifier.in_features
    if head == "single":
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            _act(act),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )
    elif head == "double":
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            _act(act),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            _act(act),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
    else:
        raise ValueError(f"Unknown head type '{head}'")
    return model


class GarbageClassifier:
    def __init__(self, name: str, ckpt_path: str, arch: str, default_img_size: int,
                 head: str = "single", act: str = "silu"):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)

        self.name = name
        self.classes = list(ckpt["class_names"])
        num_classes = int(ckpt["num_classes"])
        if len(self.classes) != num_classes:
            raise ValueError(
                f"{name}: class_names has {len(self.classes)} entries "
                f"but num_classes={num_classes}"
            )
        self.img_size = int(ckpt.get("img_size", default_img_size))

        model = build_model(arch, num_classes, head, act)
        model.load_state_dict(ckpt["model_state_dict"])
        self.model = model.to(DEVICE).eval()

        # Test-time transform: resize + normalize only (no augmentation)
        self.tf = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

        # Warm-up so the first real request is not slow
        with torch.inference_mode():
            self.model(torch.zeros(1, 3, self.img_size, self.img_size, device=DEVICE))

        logger.info("Loaded %s | head=%s act=%s classes=%s img_size=%d device=%s",
                    name, head, act, self.classes, self.img_size, DEVICE)

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict:
        x = self.tf(image.convert("RGB")).unsqueeze(0).to(DEVICE)
        probs = torch.softmax(self.model(x), dim=1)[0].cpu()
        top = int(probs.argmax())
        return {
            "predicted_class": self.classes[top],
            "confidence": round(float(probs[top]), 4),
            "probabilities": {
                c: round(float(p), 4) for c, p in zip(self.classes, probs)
            },
        }


def load_image(data: bytes, min_side: int, max_side: int) -> Image.Image:
    """Decode and validate image bytes. Raises ValueError with a clear message."""
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()                          # checks file integrity
        img = Image.open(io.BytesIO(data))    # re-open after verify()
    except Exception:
        raise ValueError("File is not a valid or readable image.")

    # Check dimensions BEFORE decoding pixels (memory protection)
    w, h = img.size
    if min(w, h) < min_side:
        raise ValueError(f"Image too small ({w}x{h}); minimum side is {min_side}px.")
    if max(w, h) > max_side:
        raise ValueError(f"Image too large ({w}x{h}); maximum side is {max_side}px.")

    try:
        img.load()
        img = ImageOps.exif_transpose(img)    # fix phone-photo rotation
    except Exception:
        raise ValueError("File is not a valid or readable image.")
    return img