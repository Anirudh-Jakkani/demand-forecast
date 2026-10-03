# One image for every service (API, worker, dashboard, MLflow server, Prefect server);
# docker-compose.yml picks the command.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    MLFLOW_DISABLE_AGENT_HINT=1

# libgomp: OpenMP runtime LightGBM links against.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

WORKDIR /app

# Dependencies first: this layer is only rebuilt when the lockfile changes.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra postgres --no-install-project

COPY src ./src
COPY configs ./configs
COPY dashboard ./dashboard
COPY docker ./docker
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra postgres

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data /app/mlartifacts \
    && chown -R app:app /app
USER app

EXPOSE 8000 8501 5000 4200
CMD ["forecast", "serve", "--host", "0.0.0.0", "--port", "8000"]
