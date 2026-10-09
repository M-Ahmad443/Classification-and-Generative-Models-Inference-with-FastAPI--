import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from PIL import Image

from app import config
from app.generator import ImageGenerator, comparison_png, to_b64
from app.inference import DEVICE, GarbageClassifier, load_image
from app.schemas import (GenerationResponse, GeneratorInfo, ModelInfo,
                         PredictionResponse)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("garbage-api")

state = {"models": {}, "generator": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Classifiers: must load, otherwise startup fails
    for name in config.ENABLED_MODELS:
        if name not in config.MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{name}'. Known: {list(config.MODEL_REGISTRY)}")
        spec = config.MODEL_REGISTRY[name]
        path = config.MODELS_DIR / spec["file"]
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        state["models"][name] = GarbageClassifier(
            name, str(path), spec["arch"], config.IMG_SIZE, spec["head"], spec["act"]
        )

    # Generator: if it fails, the classifiers keep working
    if config.ENABLE_GENERATOR:
        try:
            if not config.GENERATOR_PATH.exists():
                raise FileNotFoundError(f"Checkpoint not found: {config.GENERATOR_PATH}")
            state["generator"] = ImageGenerator(
                str(config.GENERATOR_PATH), config.GENERATOR_IMG_SIZE,
                config.GENERATOR_NORM, config.GENERATOR_NOTE, config.GENERATOR_OUTPUT,
            )
        except Exception:
            logger.exception("Generator failed to load; /generate endpoints are disabled")
    yield
    state["models"].clear()
    state["generator"] = None


app = FastAPI(
    title="Garbage Classification & Denoising API",
    description="Classify waste images with EfficientNet-B0 or DenseNet121, and denoise "
                "images with a generator model. " + config.GENERATOR_NOTE,
    version="1.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("[%s] Unhandled error on %s", request_id, request.url.path)
        return JSONResponse(status_code=500,
                            content={"detail": "Internal server error",
                                     "request_id": request_id})
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info("[%s] %s %s -> %d (%.1f ms)", request_id, request.method,
                request.url.path, response.status_code, ms)
    return response


# ---------------------------------------------------------------- ops
@app.get("/health", tags=["ops"])
def health():
    return {
        "status": "ok",
        "models_loaded": sorted(state["models"]),
        "generator_loaded": state["generator"] is not None,
    }


@app.get("/models", response_model=List[ModelInfo], tags=["ops"])
def list_models():
    return [
        ModelInfo(name=n, classes=c.classes, img_size=c.img_size, device=str(DEVICE))
        for n, c in state["models"].items()
    ]


# ---------------------------------------------------------------- shared upload handling
async def _read_image(request: Request, file: UploadFile) -> Image.Image:
    rid = request.state.request_id

    if file.content_type not in config.ALLOWED_CONTENT_TYPES:
        raise HTTPException(415, f"Unsupported type '{file.content_type}'. "
                                 f"Allowed: {sorted(config.ALLOWED_CONTENT_TYPES)}")

    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    data = await file.read(max_bytes + 1)        # read at most limit + 1 byte
    if len(data) == 0:
        raise HTTPException(400, "Empty file.")
    if len(data) > max_bytes:
        raise HTTPException(413, f"File exceeds {config.MAX_UPLOAD_MB} MB limit.")

    try:
        return load_image(data, config.MIN_IMAGE_SIDE, config.MAX_IMAGE_SIDE)
    except ValueError as e:
        logger.warning("[%s] Rejected image: %s", rid, e)
        raise HTTPException(400, str(e))


# ---------------------------------------------------------------- classification
async def _predict(request: Request, model_name: str, file: UploadFile) -> PredictionResponse:
    rid = request.state.request_id

    clf = state["models"].get(model_name)
    if clf is None:
        raise HTTPException(503, f"Model '{model_name}' is not loaded.")

    image = await _read_image(request, file)

    start = time.perf_counter()
    result = await run_in_threadpool(clf.predict, image)
    latency = round((time.perf_counter() - start) * 1000, 2)

    logger.info("[%s] model=%s prediction=%s conf=%.3f inference=%.1f ms",
                rid, model_name, result["predicted_class"], result["confidence"], latency)
    return PredictionResponse(request_id=rid, model=model_name,
                              img_size=clf.img_size, latency_ms=latency, **result)


@app.post("/predict/efficientnet_b0", response_model=PredictionResponse, tags=["inference"])
async def predict_efficientnet(request: Request, file: UploadFile = File(...)):
    return await _predict(request, "efficientnet_b0", file)


@app.post("/predict/densenet121", response_model=PredictionResponse, tags=["inference"])
async def predict_densenet(request: Request, file: UploadFile = File(...)):
    return await _predict(request, "densenet121", file)


# ---------------------------------------------------------------- generation (denoising)
async def _run_generator(request: Request, file: UploadFile):
    rid = request.state.request_id

    gen = state["generator"]
    if gen is None:
        raise HTTPException(503, "Generator model is not loaded.")

    image = await _read_image(request, file)

    start = time.perf_counter()
    inp, out = await run_in_threadpool(gen.generate, image)
    latency = round((time.perf_counter() - start) * 1000, 2)

    logger.info("[%s] generator inference=%.1f ms", rid, latency)
    return gen, inp, out, latency


@app.get("/generate/info", response_model=GeneratorInfo, tags=["generation"])
def generator_info():
    gen = state["generator"]
    if gen is None:
        raise HTTPException(503, "Generator model is not loaded.")
    return GeneratorInfo(name=gen.name, note=gen.note, img_size=gen.img_size,
                         normalization=gen.norm, device=str(DEVICE))


@app.post("/generate/denoise", response_model=GenerationResponse, tags=["generation"],
          description="Returns the input and the denoised image as base64 PNG. "
                      + config.GENERATOR_NOTE)
async def generate_denoise(request: Request, file: UploadFile = File(...)):
    gen, inp, out, latency = await _run_generator(request, file)
    return GenerationResponse(
        request_id=request.state.request_id, model=gen.name, note=gen.note,
        img_size=gen.img_size, input_image_b64=to_b64(inp),
        output_image_b64=to_b64(out), latency_ms=latency,
    )


@app.post("/generate/denoise/plot", tags=["generation"], response_class=Response,
          responses={200: {"content": {"image/png": {}}}},
          description="Returns one PNG with the input and the denoised image side by side. "
                      + config.GENERATOR_NOTE)
async def generate_denoise_plot(request: Request, file: UploadFile = File(...)):
    gen, inp, out, _ = await _run_generator(request, file)
    png = await run_in_threadpool(comparison_png, inp, out)
    return Response(content=png, media_type="image/png",
                    headers={"X-Model-Note": gen.note})