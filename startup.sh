#!/bin/bash
set -e

# Run ingestion only if vector store doesn't exist yet
if [ ! -d "/app/chroma_langchain_db" ]; then
    echo "First run: ingesting documents..."
    python -c "from core.ingestion import ingest; ingest()"
fi

exec streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.fileWatcherType=none \
    --browser.gatherUsageStats=false
