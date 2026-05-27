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
COPY startup.sh .
RUN chmod +x startup.sh

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["./startup.sh"]
