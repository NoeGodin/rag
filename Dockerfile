FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.deploy.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py .
COPY app.py .
COPY core/ core/
COPY utils/ utils/
COPY assets/ assets/

# Ingest at build time so the vector store is baked into the image
ARG FAL_KEY
ENV FAL_KEY=$FAL_KEY
RUN python -c "from core.ingestion import ingest; ingest()"

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.fileWatcherType=none", \
    "--browser.gatherUsageStats=false"]
