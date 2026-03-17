# Legal Edu RAG (Retrieval Only)

Du an hien tai da duoc don sach theo huong retrieval-only.

Giữ lại:
- Nap du lieu luat tu `data/final`
- Build embedding + FAISS index
- Truy xuat top-k dieu khoan lien quan
- Streamlit UI de kiem tra retrieval quality
- Logging va retrieval trace

Da loai bo:
- Tat ca thanh phan sinh cau tra loi bang model LLM
- Ollama service, script pull model, cache model

## 1. Chay Nhanh

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
python scripts/build_faiss_index.py
streamlit run src/app/ui/streamlit_app.py
```

Mo UI: `http://localhost:8501`

## 2. Chay Bang Docker

```bash
docker compose up -d --build
```

Lan sau khong can build lai:

```bash
docker compose up -d
```

## 3. Logging

- Runtime log: `logs/app.log`
- Retrieval trace: `logs/retrieval_trace.log`

Moi trace gom:
- timestamp
- question
- elapsed_ms
- danh sach top-k chunks (source/article/score/text)