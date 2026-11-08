"""VisionGuard Flask web app (Tasks 4, 5, 8).

All settings come from environment variables (Task 4.4), so the same image runs
in dev, Compose and the cloud without code changes:

    FLASK_SECRET_KEY, MAX_UPLOAD_MB, DATABASE_URL,
    INFERENCE_BACKEND (local|remote), MODEL_PATH, THRESHOLD, MODEL_API_URL, MODEL_API_KEY
"""

from __future__ import annotations

import base64
import io
import logging
import os
from datetime import UTC, datetime

from flask import Flask, jsonify, redirect, render_template, request, url_for
from PIL import Image
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from visionguard.inference import DISPLAY
from visionguard.preprocess import to_rgb
from visionguard.validation import UploadError, validate_upload
from visionguard.web import db
from visionguard.web.backends import BackendError, LocalBackend, RemoteBackend

log = logging.getLogger("visionguard")


def make_backend(cfg: dict):
    if cfg["INFERENCE_BACKEND"] == "remote":
        return RemoteBackend(cfg["MODEL_API_URL"], cfg["MODEL_API_KEY"])
    thr = cfg["THRESHOLD"]
    return LocalBackend(cfg["MODEL_PATH"], float(thr) if thr else None)


def thumbnail_data_uri(data: bytes, size: int = 320) -> str:
    """Small in-memory preview for the result page - nothing is written to disk."""
    img = to_rgb(Image.open(io.BytesIO(data)))
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def create_app(overrides: dict | None = None, backend=None) -> Flask:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = Flask(__name__)
    env = os.environ.get
    app.config.update(
        SECRET_KEY=env("FLASK_SECRET_KEY", "dev-only-not-secret"),
        MAX_UPLOAD_MB=float(env("MAX_UPLOAD_MB", "8")),
        DATABASE_URL=env("DATABASE_URL", "sqlite:///instance/visionguard.db"),
        INFERENCE_BACKEND=env("INFERENCE_BACKEND", "local"),
        MODEL_PATH=env("MODEL_PATH", "models/champion.keras"),
        THRESHOLD=env("THRESHOLD", ""),
        MODEL_API_URL=env("MODEL_API_URL", ""),
        MODEL_API_KEY=env("MODEL_API_KEY", ""),
    )
    app.config.update(overrides or {})
    # Werkzeug rejects bigger bodies with 413 before our code reads them.
    app.config["MAX_CONTENT_LENGTH"] = int(app.config["MAX_UPLOAD_MB"] * 1024 * 1024)

    # Start-up: connect to the DB and load the model ONCE. Fail fast with a clear log line.
    engine = db.make_engine(app.config["DATABASE_URL"])
    db.init_db(engine)
    log.info("database ready: %s", engine.url.render_as_string(hide_password=True))
    backend = backend or make_backend(app.config)
    log.info("inference backend ready: %s", backend.info())
    app.extensions["vg_engine"], app.extensions["vg_backend"] = engine, backend

    @app.context_processor
    def _globals():
        return {"DISPLAY": DISPLAY, "max_mb": app.config["MAX_UPLOAD_MB"], "backend_name": backend.name}

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/predict")
    def predict():
        up = validate_upload(request.files.get("image"))
        try:
            res = backend.predict(up.data, up.filename)
        except BackendError as exc:
            log.error("prediction failed: %s", exc)
            return render_template("error.html", title="Model unavailable", status=502,
                                   message="The model service did not answer. Please try again in a minute."), 502
        row_id = db.record(
            engine, filename=up.filename, image_sha256=up.sha256, image_width=up.width, image_height=up.height,
            label=res["label"], confidence=res["confidence"], prob_ai=res["prob_ai"], threshold=res["threshold"],
            model_name=res["model_name"], model_version=str(res["model_version"]), backend=backend.name,
            latency_ms=res.get("latency_ms"),
        )
        log.info("prediction #%s %s -> %s (%.3f)", row_id, up.filename, res["label"], res["confidence"])
        return render_template("result.html", r=res, up=up, row_id=row_id, preview=thumbnail_data_uri(up.data))

    @app.get("/history")
    def history():
        page = max(1, request.args.get("page", 1, type=int))
        per = 25
        rows, total = db.history(engine, per, (page - 1) * per)
        return render_template("history.html", rows=rows, total=total, page=page,
                               pages=max(1, -(-total // per)), stats=db.stats(engine))

    @app.get("/api/history")
    def api_history():
        rows, total = db.history(engine, request.args.get("limit", 50, type=int))
        for r in rows:
            r["created_at"] = r["created_at"].isoformat()
        return jsonify(total=total, items=rows)

    @app.get("/health")
    def health():
        info = backend.info()
        db_ok = db.ping(engine)
        ok = db_ok and info.get("model_loaded", False)
        # backend info first, so this app's own fields (status, upload limit) win over the API's
        body = {**info, "status": "ok" if ok else "degraded", "database": "ok" if db_ok else "down",
                "time": datetime.now(UTC).isoformat(), "debug": app.debug,
                "max_upload_mb": app.config["MAX_UPLOAD_MB"]}
        return jsonify(body), 200 if ok else 503

    # ---------- friendly error pages ----------
    @app.errorhandler(UploadError)
    def _upload_error(exc: UploadError):
        return render_template("error.html", title=exc.title, message=str(exc), status=exc.status), exc.status

    @app.errorhandler(RequestEntityTooLarge)
    def _too_large(_exc):
        return render_template("error.html", title="File too large", status=413,
                               message=f"Images must be at most {app.config['MAX_UPLOAD_MB']:g} MB."), 413

    @app.errorhandler(HTTPException)
    def _http(exc: HTTPException):
        if request.path.startswith(("/api/", "/health")):
            return jsonify(error=exc.description), exc.code
        return render_template("error.html", title=exc.name, message=exc.description, status=exc.code), exc.code

    @app.errorhandler(Exception)
    def _unexpected(exc: Exception):
        log.exception("unhandled error")
        return render_template("error.html", title="Something went wrong", status=500,
                               message="An unexpected error occurred. It has been logged."), 500

    @app.get("/predict")
    def predict_get():
        return redirect(url_for("index"))

    return app
