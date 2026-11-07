"""Model API tests with a stub predictor (no TensorFlow)."""

import io

import pytest
from PIL import Image

from visionguard.api import create_api
from visionguard.inference import Prediction


class StubPredictor:
    model_name, model_version, threshold, img_size = "stub", "1", 0.5, 224

    def predict_bytes(self, data):
        return Prediction("real", 0.8, 0.2, 0.5, "stub", "1", 2.0)


@pytest.fixture()
def client():
    return create_api(StubPredictor(), api_key="s3cret").test_client()


def img():
    b = io.BytesIO()
    Image.new("RGB", (32, 32)).save(b, "JPEG")
    return b.getvalue()


def test_health_open(client):
    assert client.get("/health").get_json()["status"] == "ok"


def test_key_required(client):
    r = client.post("/predict", data={"image": (io.BytesIO(img()), "a.jpg")})
    assert r.status_code == 401


def test_predict_ok(client):
    r = client.post("/predict", data={"image": (io.BytesIO(img()), "a.jpg")}, headers={"X-API-Key": "s3cret"})
    assert r.status_code == 200
    assert r.get_json()["label"] == "real"


def test_bad_image_json_error(client):
    r = client.post("/predict", data={"image": (io.BytesIO(b"xx"), "a.png")}, headers={"X-API-Key": "s3cret"})
    assert r.status_code == 400 and "error" in r.get_json()
