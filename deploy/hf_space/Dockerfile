# VisionGuard model API for Hugging Face Spaces (Docker SDK). Task 7.
# HF builds this image itself from the Space repo; the champion model is BUILT IN
# (the Space has no shared disk to mount from), so each deployment is immutable:
# image = code + exact model version.
FROM python:3.11-slim
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 MODEL_PATH=/app/models/champion.keras PORT=7860
# HF Spaces run containers as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY --chown=user visionguard ./visionguard
COPY --chown=user models ./models
USER user
EXPOSE 7860
# MODEL_API_KEY is set as a Space *secret* in the HF settings page, never in this file.
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "120", \
     "--access-logfile", "-", "visionguard.api.wsgi:app"]
