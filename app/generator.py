import base64
import io
import logging
from typing import Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from app.generator_net import build_generator
from app.inference import DEVICE, MEAN, STD

logger = logging.getLogger("garbage-api")

NORMS = {
    "none": (None, None),
    "half": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    "imagenet": (MEAN, STD),
}


def _extract_state_dict(ckpt):
    """Accept a raw state_dict or one wrapped under a common key."""
    if isinstance(ckpt, dict):
        for key in ("model_state", "model_state_dict", "generator_state_dict",
                    "state_dict", "generator", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def comparison_png(inp: Image.Image, out: Image.Image,
                   titles=("Input (noisy)", "Generated (denoised)")) -> bytes:
    """Side-by-side plot of input and generated image, returned as PNG bytes."""
    scale = max(1, 384 // inp.width)          # enlarge small images for display
    if scale > 1:
        inp = inp.resize((inp.width * scale, inp.height * scale), Image.NEAREST)
        out = out.resize((out.width * scale, out.height * scale), Image.NEAREST)

    pad, title_h = 12, 32
    w, h = inp.size
    canvas = Image.new("RGB", (w * 2 + pad * 3, h + title_h + pad), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)

    for i, (img, title) in enumerate(zip((inp, out), titles)):
        x = pad + i * (w + pad)
        draw.text((x, 8), title, fill="black", font=font)
        canvas.paste(img, (x, title_h))

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


class ImageGenerator:
    def __init__(self, ckpt_path: str, default_img_size: int, norm: str, note: str,
                 output: str = "01"):
        if norm not in NORMS:
            raise ValueError(f"GENERATOR_NORM must be one of {list(NORMS)}, got '{norm}'")
        if output not in ("01", "tanh"):
            raise ValueError("GENERATOR_OUTPUT must be '01' or 'tanh'")

        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        self.name = "generator_best_model"
        self.note = note
        self.norm = norm
        self.output = output
        # The ViT branch has a fixed 14x14 positional embedding, so 224 is required
        self.img_size = default_img_size

        state = _extract_state_dict(ckpt)
        state = {k.removeprefix("module."): v for k, v in state.items()}

        model = build_generator()
        model.load_state_dict(state)
        self.model = model.to(DEVICE).eval()

        # Same transformation as the test set: Resize(224,224) + ToTensor
        mean, std = NORMS[norm]
        self.resize = transforms.Resize((self.img_size, self.img_size))
        ops = [self.resize, transforms.ToTensor()]
        if mean is not None:
            ops.append(transforms.Normalize(mean, std))
        self.tf = transforms.Compose(ops)
        self.mean = None if mean is None else torch.tensor(mean).view(3, 1, 1)
        self.std = None if std is None else torch.tensor(std).view(3, 1, 1)

        # Warm-up so the first real request is not slow
        with torch.inference_mode():
            self.model(torch.zeros(1, 3, self.img_size, self.img_size, device=DEVICE))

        logger.info("Loaded generator | img_size=%d norm=%s output=%s device=%s",
                    self.img_size, norm, output, DEVICE)

    @torch.inference_mode()
    def generate(self, image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        rgb = image.convert("RGB")
        x = self.tf(rgb).unsqueeze(0).to(DEVICE)

        y = self.model(x)
        if isinstance(y, (tuple, list)):          # (fused, cnn_out, vit_out) -> fused
            y = y[0]
        y = y.squeeze(0).float().cpu()
        logger.info("generator raw output range: min=%.3f max=%.3f",
                    float(y.min()), float(y.max()))

        if self.output == "tanh":                 # -1..1 -> 0..1
            y = (y + 1) / 2
        elif self.mean is not None:               # output is in the normalized space
            y = y * self.std + self.mean
        y = y.clamp(0, 1)

        return self.resize(rgb), transforms.ToPILImage()(y)