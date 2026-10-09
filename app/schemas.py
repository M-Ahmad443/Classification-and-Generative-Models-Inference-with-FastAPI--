from typing import Dict, List

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    request_id: str
    model: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    img_size: int
    latency_ms: float


class ModelInfo(BaseModel):
    name: str
    classes: List[str]
    img_size: int
    device: str


class GenerationResponse(BaseModel):
    request_id: str
    model: str
    note: str
    img_size: int
    input_image_b64: str      # PNG, base64: the input as the model sees it
    output_image_b64: str     # PNG, base64: the generated (denoised) image
    latency_ms: float


class GeneratorInfo(BaseModel):
    name: str
    note: str
    img_size: int
    normalization: str
    device: str