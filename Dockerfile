# VisionGuard web app - optimised image (Task 6.2 / 6.4)
# Build:  docker build -t 24l-0672_a2_ops-web .

# ---------- stage 1: build wheels/venv (has pip & build tools, thrown away) ----------
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
# Copy ONLY the requirements first: this layer (the slow ~2 min TensorFlow install) is
# cached and reused as long as requirements-serve.txt does not change, even if the code does.
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt \
 && find /opt/venv -name "__pycache__" -prune -exec rm -rf {} + \
 && rm -rf /opt/venv/lib/python3.11/site-packages/tensorflow/include

# ---------- stage 2: runtime ----------
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH=/opt/venv/bin:$PATH \
    TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1 \
    MODEL_PATH=/app/models/champion.keras PORT=5000
# Unprivileged user: if the app is compromised the attacker is not root in the container.
RUN groupadd --system app && useradd --system --gid app --home /app --shell /usr/sbin/nologin app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
# Source code last: it changes most often, so only these cheap layers rebuild.
COPY --chown=app:app visionguard ./visionguard
COPY --chown=app:app gunicorn.conf.py .
USER app
EXPOSE 5000
# The model is NOT baked in: it is mounted read-only at /app/models (see docker-compose.yml).
# No secrets here either: they arrive at runtime from the untracked .env via env_file.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=4).status == 200 else 1)"
CMD ["gunicorn", "-c", "gunicorn.conf.py", "visionguard.web.wsgi:app"]
