# Production WSGI server settings (used inside Docker).
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
# One worker process = one copy of the TensorFlow model in RAM; threads handle concurrency
# (TF releases the GIL during inference).
workers = int(os.environ.get("GUNICORN_WORKERS", "1"))
threads = int(os.environ.get("GUNICORN_THREADS", "4"))
worker_class = "gthread"
timeout = 120          # first request + model warm-up can be slow on small CPUs
accesslog = "-"        # access log to stdout -> visible in `docker compose logs`
errorlog = "-"
loglevel = "info"
