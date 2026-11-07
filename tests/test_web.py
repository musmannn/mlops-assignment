"""Web app tests: validation failure cases, DB trail, health. No TensorFlow needed."""

import io

import pytest
from PIL import Image

from visionguard.web import create_app
from visionguard.web.backends import FakeBackend


def png_bytes(size=(64, 48), color=(200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path):
    app = create_app({"DATABASE_URL": f"sqlite:///{tmp_path}/t.db", "MAX_UPLOAD_MB": 1, "TESTING": True},
                     backend=FakeBackend(prob_ai=0.9))
    return app.test_client()


def upload(client, data, name):
    return client.post("/predict", data={"image": (io.BytesIO(data), name)}, content_type="multipart/form-data")


def test_index(client):
    r = client.get("/")
    assert r.status_code == 200 and b"Analyse image" in r.data


def test_prediction_is_logged(client):
    r = upload(client, png_bytes(), "cat.png")
    assert r.status_code == 200
    assert b"AI-generated" in r.data and b"90.0%" in r.data
    h = client.get("/api/history").get_json()
    assert h["total"] == 1
    row = h["items"][0]
    assert row["label"] == "ai" and row["model_name"] == "fake" and row["filename"] == "cat.png"
    assert len(row["image_sha256"]) == 64
    assert b"cat.png" in client.get("/history").data


def test_wrong_extension_rejected(client):
    r = upload(client, b"hello", "notes.txt")
    assert r.status_code == 415 and b"Unsupported file type" in r.data


def test_fake_image_rejected(client):
    r = upload(client, b"definitely not a jpeg" * 50, "photo.jpg")
    assert r.status_code == 400 and b"could not be decoded" in r.data


def test_too_large_rejected(client):
    r = upload(client, b"0" * (2 * 1024 * 1024), "big.png")
    assert r.status_code == 413 and b"File too large" in r.data


def test_no_file(client):
    r = client.post("/predict", data={}, content_type="multipart/form-data")
    assert r.status_code == 400


def test_filename_sanitized(client):
    r = upload(client, png_bytes(), "../../etc/passwd.png")
    assert r.status_code == 200
    assert client.get("/api/history").get_json()["items"][0]["filename"] == "etc_passwd.png"


def test_health(client):
    r = client.get("/health")
    body = r.get_json()
    assert r.status_code == 200 and body["status"] == "ok" and body["database"] == "ok"
    assert body["debug"] is False


def test_nothing_written_to_disk(client, tmp_path):
    before = set(tmp_path.rglob("*"))
    upload(client, png_bytes(), "x.png")
    after = set(tmp_path.rglob("*"))
    assert {p.suffix for p in after - before} <= {".db", ".db-journal", ""}
