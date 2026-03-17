FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PROJECT_ROOT=/app \
    DATA_DIR=/app/data/final \
    INDEX_DIR=/app/outputs/indexes/faiss \
    LOGS_DIR=/app/logs

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY src /app/src
COPY scripts /app/scripts
COPY README.md /app/README.md
RUN chmod +x /app/scripts/run_streamlit.sh

ENV PYTHONPATH=/app/src
EXPOSE 8501

CMD ["/app/scripts/run_streamlit.sh"]
