import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import config
from app.main import app

ENDPOINTS = ["efficientnet_b0", "densenet121"]


@pytest.fixture(scope="module")
def client():
    missing = [config.MODELS_DIR / config.MODEL_REGISTRY[m]["file"]
               for m in config.ENABLED_MODELS
               if not (config.MODELS_DIR / config.MODEL_REGISTRY[m]["file"]).exists()]
    if missing:
        pytest.skip(f"Checkpoint files missing: {missing}")
    with TestClient(app) as c:          # runs lifespan -> loads models
        yield c


def make_image(size=(200, 200), fmt="PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 80, 200)).save(buf, format=fmt)
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert set(r.json()["models_loaded"]) == set(config.ENABLED_MODELS)


def test_models_list(client):
    r = client.get("/models")
    assert r.status_code == 200
    assert {m["name"] for m in r.json()} == set(config.ENABLED_MODELS)


@pytest.mark.parametrize("name", ENDPOINTS)
def test_predict_ok(client, name):
    r = client.post(f"/predict/{name}",
                    files={"file": ("a.png", make_image(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == name
    assert body["predicted_class"] in body["probabilities"]
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-2
    assert "x-request-id" in r.headers


@pytest.mark.parametrize("name", ENDPOINTS)
def test_wrong_content_type(client, name):
    r = client.post(f"/predict/{name}", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 415


@pytest.mark.parametrize("name", ENDPOINTS)
def test_empty_file(client, name):
    r = client.post(f"/predict/{name}", files={"file": ("a.png", b"", "image/png")})
    assert r.status_code == 400


@pytest.mark.parametrize("name", ENDPOINTS)
def test_corrupt_image(client, name):
    r = client.post(f"/predict/{name}",
                    files={"file": ("a.png", b"not an image", "image/png")})
    assert r.status_code == 400


@pytest.mark.parametrize("name", ENDPOINTS)
def test_image_too_small(client, name):
    r = client.post(f"/predict/{name}",
                    files={"file": ("a.png", make_image((10, 10)), "image/png")})
    assert r.status_code == 400


@pytest.mark.parametrize("name", ENDPOINTS)
def test_file_too_large(client, name):
    big = b"0" * (config.MAX_UPLOAD_MB * 1024 * 1024 + 1)
    r = client.post(f"/predict/{name}", files={"file": ("a.png", big, "image/png")})
    assert r.status_code == 413