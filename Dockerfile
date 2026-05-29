FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project

COPY config.py .
COPY app.py .
COPY chainlit.md .
COPY .chainlit/ .chainlit/
COPY public/ public/
COPY core/ core/
COPY utils/ utils/

EXPOSE 8000

HEALTHCHECK CMD curl --fail http://localhost:8000/ || exit 1

ENTRYPOINT ["uv", "run", "chainlit", "run", "app.py", \
    "--host", "0.0.0.0", \
    "--port", "8000"]
