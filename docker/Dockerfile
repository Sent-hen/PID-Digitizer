# CPU image for the P&ID Digitizer service (FastAPI + review UI).
# Build from the repository root:  docker build -f docker/Dockerfile -t pid-digitizer .
# Weights are not baked in: mount weights/ (see docker-compose.yml, scripts/fetch_weights.py).

ARG PYTHON=3.13

# ---- builder: resolve pinned dependencies into a venv -----------------------------------
FROM python:${PYTHON}-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install uv==0.12.22
WORKDIR /build
COPY pyproject.toml uv.lock ./
# Locked runtime dependencies (no dev tools, no CUDA torch group) ...
RUN uv export --frozen --no-dev --no-group ml --no-emit-project --no-hashes -o requirements.txt \
 && python -m venv /opt/venv \
 && /opt/venv/bin/pip install -r requirements.txt \
 # ... plus the CPU build of the same torch/torchvision versions as uv.lock.
 && /opt/venv/bin/pip install torch==2.14.1 torchvision==0.29.1 \
        --index-url https://download.pytorch.org/whl/cpu

# ---- runtime -------------------------------------------------------------------------------
FROM python:${PYTHON}-slim
RUN useradd --create-home --uid 10001 app \
 && mkdir -p /data /app/weights && chown app:app /data
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY configs ./configs
COPY src ./src
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PID_DATA_DIR=/data \
    PID_DEVICE=cpu \
    OMP_NUM_THREADS=4
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=5 \
  CMD python -c "import urllib.request,json,sys; h=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)); sys.exit(0 if h['model_loaded'] else 1)"
CMD ["uvicorn", "pid.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
