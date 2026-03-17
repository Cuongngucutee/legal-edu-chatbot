#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f "outputs/indexes/faiss/legal_faiss.index" ]]; then
  echo "[startup] FAISS index not found. Building index first..."
  python scripts/build_faiss_index.py
fi

exec streamlit run src/app/ui/streamlit_app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.fileWatcherType none
