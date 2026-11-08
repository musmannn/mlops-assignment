"""Vercel entry point: the VisionGuard model API as a Python serverless function.

Vercel's Python runtime serves the WSGI object named `app`. The model is the ONNX export of
the champion (onnxruntime instead of TensorFlow, which is far over Vercel's 500 MB limit).
It is loaded once per warm instance, at import time, never per request.
"""

import os
import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlencode

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MODEL_PATH", str(ROOT / "models" / "champion.onnx"))
os.environ.setdefault("MAX_UPLOAD_MB", "4")  # Vercel rejects request bodies over 4.5 MB

from visionguard.api import create_api  # noqa: E402

flask_app = create_api()


def app(environ, start_response):
    """Undo Vercel's rewrite.

    vercel.json rewrites every URL to /api/index?__path=/<original>, because the function is a
    single file. Here the original path is put back into PATH_INFO so Flask's normal routes
    (/health, /predict) match, and the helper parameter is removed from the query string.
    """
    query = parse_qsl(environ.get("QUERY_STRING", ""), keep_blank_values=True)
    original = next((v for k, v in query if k == "__path"), None)
    if original is not None:
        environ["PATH_INFO"] = original or "/"
        environ["QUERY_STRING"] = urlencode([(k, v) for k, v in query if k != "__path"])
    print(f"{environ.get('REQUEST_METHOD')} {environ['PATH_INFO']}", file=sys.stderr)
    return flask_app(environ, start_response)
