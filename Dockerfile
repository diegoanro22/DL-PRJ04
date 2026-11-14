# Imagen de la API + panel del agente (CPU, ONNX Runtime). Pensada para Hugging Face Spaces
# (sdk: docker, puerto 7860) pero corre igual en cualquier lado:
#   docker build -t verificador-voz . && docker run -p 7860:7860 verificador-voz
# Los modelos se copian de app/models/<nombre>/ (generados con python -m app.package_model).

# 1) Panel: React + Vite -> app/static
FROM node:22-slim AS web
RUN corepack enable
WORKDIR /build/app/web
COPY app/web/package.json app/web/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY app/web/ ./
RUN pnpm build

# 2) API: sin PyTorch, solo lo necesario para inferir
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 app
WORKDIR /home/app/srv
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt && chown app:app /home/app/srv

USER app
COPY --chown=app:app app/__init__.py app/api.py app/inference.py app/monitoring.py app/
COPY --chown=app:app --from=web /build/app/static app/static
COPY --chown=app:app app/models app/models

ENV MODELS_DIR=/home/app/srv/app/models \
    DEFAULT_MODEL=C0 \
    ONNX_THREADS=2 \
    PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT}"]
