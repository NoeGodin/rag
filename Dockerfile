FROM python:3.13-slim

# Copier uv depuis l'image officielle pour des builds ultra-rapides
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de déclaration de dépendances uv
COPY pyproject.toml uv.lock ./

# Synchroniser les dépendances (crée implicitement l'environnement virtuel /app/.venv)
RUN uv sync --frozen --no-install-project

COPY config.py .
COPY app.py .
COPY chainlit.md .
COPY .chainlit/ .chainlit/
COPY public/ public/
COPY core/ core/
COPY utils/ utils/
COPY assets/ assets/

# Ingest at build time so the vector store is baked into the image
ARG FAL_KEY
ENV FAL_KEY=$FAL_KEY
RUN uv run python -c "from core.ingestion import ingest; ingest()"

EXPOSE 8000

HEALTHCHECK CMD curl --fail http://localhost:8000/ || exit 1

ENTRYPOINT ["uv", "run", "chainlit", "run", "app.py", \
    "--host", "0.0.0.0", \
    "--port", "8000"]
