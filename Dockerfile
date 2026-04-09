FROM python:3.11-slim

# Metadata
LABEL maintainer="supply-chain-openenv"
LABEL org.opencontainers.image.description="Supply Chain Disruption Manager — OpenEnv Environment"

# HF Spaces runs as non-root user 1000
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY models.py .
COPY env.py .
COPY app.py .
COPY inference.py .
COPY openenv.yaml .

COPY tasks/ ./tasks/
COPY graders/ ./graders/

# HF Spaces expects port 7860
ENV PORT=7860
EXPOSE 7860

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; r = requests.get('http://localhost:7860/health'); exit(0 if r.status_code == 200 else 1)"

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
