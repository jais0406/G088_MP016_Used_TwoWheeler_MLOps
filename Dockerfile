# Prediction service for MP016 - Used Two-Wheeler Price (Lucknow)
# Build:  docker build -t mp016-api:1.0 .
# Run:    docker run --rm -p 8000:8000 mp016-api:1.0
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=models/model.joblib \
    MODEL_VERSION=champion

WORKDIR /app

# Install dependencies first so this layer is cached when only code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Then copy the service code and the champion model it loads at startup.
COPY app/ ./app/
COPY models/model.joblib ./models/model.joblib

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
