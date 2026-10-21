# syntax=docker/dockerfile:1
FROM python:3.12.12-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CTP_CONFIGS_DIR=/app/configs \
    CTP_STORAGE_ROOT=/data/objects \
    CTP_DATABASE_URL=sqlite:////data/ctpipeline.db

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[postgres]"
COPY configs ./configs

ARG CODE_REVISION=unknown
ENV CTP_CODE_REVISION=${CODE_REVISION}

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin ctp \
    && mkdir -p /data && chown ctp:ctp /data
USER 10001
VOLUME ["/data"]
EXPOSE 8000

# Default: API. Workers override the command with ["ctpipeline-worker"].
CMD ["uvicorn", "ctpipeline.api.app:build_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
