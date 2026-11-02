# LLM Proxy Service Dockerfile
# Build from repo root:
#   docker build -f services/llm-proxy/Dockerfile -t aiv2/llm-proxy:dev .
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including Tesseract OCR with ALL languages
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-all \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY services/llm-proxy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY services/llm-proxy/src/ ./src/
COPY packages/aiv2-model-contract/ /app/packages/aiv2-model-contract/

ENV PYTHONPATH=/app:/app/packages/aiv2-model-contract/src

# Expose port
EXPOSE 8005

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8005/health')" || exit 1

# Run service
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8005"]
