"""VisionGuard model API (Task 7): JSON in, JSON out, deployed to Hugging Face Spaces.

POST /predict   multipart field "image", header "X-API-Key: <MODEL_API_KEY>"
GET  /health    no key needed; reports model version and threshold
"""

from __future__ import annotations

import hmac
import logging
import os
from datetime import UTC, datetime

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from visionguard.validation import UploadError, validate_upload

log = logging.getLogger("visionguard.api")


def create_api(predictor=None, api_key: str | None = None) -> Flask:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = Flask(__name__)
    max_mb = float(os.environ.get("MAX_UPLOAD_MB", "8"))
    app.config["MAX_CONTENT_LENGTH"] = int(max_mb * 1024 * 1024)
    key = api_key if api_key is not None else os.environ.get("MODEL_API_KEY", "")
    if not key:
        log.warning("MODEL_API_KEY is empty - the API is OPEN to anyone")

    if predictor is None:
        from visionguard.inference import predictor_from_env

        predictor = predictor_from_env()  # loaded once at start-up
    log.info("model %s v%s loaded (threshold %.3f)", predictor.model_name, predictor.model_version, predictor.threshold)
    started = datetime.now(UTC)
    counter = {"n": 0}

    @app.before_request
    def _auth():
        if request.endpoint in ("predict",) and key:
            given = request.headers.get("X-API-Key", "")
            # constant-time comparison: no timing side channel on the secret
            if not hmac.compare_digest(given.encode(), key.encode()):
                return jsonify(error="missing or wrong X-API-Key"), 401
        return None

    @app.get("/")
    @app.get("/health")
    def health():
        return jsonify(status="ok", service="visionguard-model-api", model_loaded=True,
                       model_name=predictor.model_name, model_version=predictor.model_version,
                       threshold=predictor.threshold, img_size=predictor.img_size,
                       requests_served=counter["n"], up_since=started.isoformat(), max_upload_mb=max_mb)

    @app.post("/predict")
    def predict():
        up = validate_upload(request.files.get("image"))
        res = predictor.predict_bytes(up.data).to_dict()
        counter["n"] += 1
        log.info("predict %s %dx%d -> %s %.3f (%s ms)", up.filename, up.width, up.height,
                 res["label"], res["confidence"], res["latency_ms"])
        return jsonify(res | {"filename": up.filename, "width": up.width, "height": up.height,
                              "served_at": datetime.now(UTC).isoformat()})

    @app.errorhandler(UploadError)
    def _bad(exc: UploadError):
        return jsonify(error=str(exc), title=exc.title), exc.status

    @app.errorhandler(RequestEntityTooLarge)
    def _big(_):
        return jsonify(error=f"file larger than {max_mb:g} MB"), 413

    @app.errorhandler(HTTPException)
    def _http(exc: HTTPException):
        return jsonify(error=exc.description), exc.code

    return app
