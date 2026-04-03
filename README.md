# 📚 BookRAG – Hệ thống Hỏi đáp Luật Giáo dục Việt Nam

<p align="center">
  <strong>Pipeline RAG phân cấp kết hợp Knowledge Graph cho truy vấn pháp luật giáo dục</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/framework-BookRAG-green.svg" alt="BookRAG">
  <img src="https://img.shields.io/badge/recall@K-85.4%25-brightgreen.svg" alt="Recall">
  <img src="https://img.shields.io/badge/keyword_coverage-95.8%25-brightgreen.svg" alt="Coverage">
</p>

---

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Kiến trúc](#-kiến-trúc)
- [Cài đặt](#-cài-đặt)
- [Cấu hình](#-cấu-hình)
- [Sử dụng](#-sử-dụng)
- [Benchmark](#-benchmark)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Chi tiết kỹ thuật](#-chi-tiết-kỹ-thuật)

---

## 🎯 Giới thiệu

BookRAG là pipeline truy xuất và sinh câu trả lời (RAG) chuyên biệt cho **Luật Giáo dục Việt Nam**, được xây dựng theo kiến trúc BookRAG với:

- **Hierarchical Tree Index** – Cây phân cấp 6 tầng: Corpus → Luật → Chương → Mục → Điều → Khoản → Điểm
- **Knowledge Graph** – Đồ thị tri thức với 5,500+ entities và 12,500+ quan hệ
- **Hybrid Retrieval** – Kết hợp Dense (Qdrant) + Sparse (BM25) + KG traversal
- **Cross-Encoder Reranking** – Tái xếp hạng kết quả để tăng precision
- **Chain-of-Thought Reasoning** – Suy luận 4 bước trước khi sinh câu trả lời
- **Automatic Citation** – Tự động trích dẫn nguồn (Luật, Điều, Khoản, Điểm)

### Dữ liệu

8 văn bản luật giáo dục Việt Nam đã được chunking sẵn:

| STT | Văn bản | Mô tả |
|-----|---------|-------|
| 1 | Luật 08/2012/QH13 | Luật Giáo dục đại học |
| 2 | Luật 34/2018/QH14 | Sửa đổi Luật GD đại học |
| 3 | Luật 43/2019/QH14 | Luật Giáo dục |
| 4 | Luật 74/2014/QH13 | Luật Giáo dục nghề nghiệp |
| 5 | Luật 14/2012/QH13 | Luật Giáo dục |
| 6 | Luật 30/2013/QH13 | Luật Giáo dục quốc phòng |
| 7 | Luật 73/2025/QH15 | Luật Nhà giáo |
| 8 | Luật 123/2025/QH15 | Luật GD đại học (mới) |

---

## 🏗 Kiến trúc

```
┌─────────────────────────────────────────────────────────┐
│                     User Query                          │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│  1. Query Processing                                    │
│     • Intent Classification (5 loại)                    │
│     • Query Normalization (viết tắt, chính tả)         │
│     • Named Entity Recognition (luật, điều, khoản)     │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│  2. Hybrid Retrieval (8 bước)                           │
│     ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│     │  Dense   │  │  Sparse  │  │    KG    │          │
│     │ (Qdrant) │  │ (BM25)   │  │(NetworkX)│          │
│     └────┬─────┘  └────┬─────┘  └────┬─────┘          │
│          └──────────────┼─────────────┘                 │
│                         ▼                               │
│     RRF Fusion → Intent Boost → Dedup → Rerank         │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│  3. Chain-of-Thought Reasoning (4 bước)                 │
│     • Phân tích câu hỏi                                │
│     • Lập kế hoạch                                      │
│     • Đối chiếu bằng chứng                              │
│     • Phán đoán & kiểm chứng                            │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│  4. Answer Generation                                   │
│     • LLM generation với legal prompt                   │
│     • Citation extraction                               │
│     • Confidence scoring                                │
│     • Related questions suggestion                      │
└─────────────────┬───────────────────────────────────────┘
                  ▼
┌─────────────────────────────────────────────────────────┐
│  5. Formatted Answer + Citations                        │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Cài đặt

### Yêu cầu hệ thống

- **Python** >= 3.10
- **RAM** >= 4GB (embedding model ~470MB + in-memory indexes)
- **Disk** >= 2GB (model cache)

### Bước 1: Clone & tạo môi trường

```bash
# Clone repository
git clone <repo-url>
cd testBookRAG

# Tạo conda environment (khuyến nghị)
conda create -n bookrag python=3.10 -y
conda activate bookrag

# Hoặc dùng venv
python -m venv venv
source venv/bin/activate  # Mac/Linux
```

### Bước 2: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

> **Lưu ý:** Lần đầu chạy, embedding model `paraphrase-multilingual-MiniLM-L12-v2` (~470MB) sẽ được tự động tải từ HuggingFace.

### Bước 3: Cấu hình API key (tuỳ chọn)

```bash
cp .env.example .env
```

Mở file `.env` và thêm API key:

```env
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4o-mini
```

> **Quan trọng:** API key **không bắt buộc**. Nếu không có, pipeline vẫn hoạt động ở **fallback mode** — trả về trích dẫn trực tiếp từ luật thay vì câu trả lời được tổng hợp bởi LLM.

---

## ⚙ Cấu hình

Tất cả cấu hình nằm trong `src/config.py`:

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `OPENAI_MODEL` | `gpt-4o-mini` | Model LLM cho reasoning & generation |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Model embedding |
| `DENSE_TOP_K` | `20` | Số kết quả từ dense retrieval |
| `SPARSE_TOP_K` | `20` | Số kết quả từ BM25 |
| `RERANK_TOP_K` | `10` | Số kết quả sau reranking |
| `MAX_CONTEXT_CHUNKS` | `8` | Số chunks tối đa đưa vào LLM |
| `RRF_K` | `60` | Hằng số Reciprocal Rank Fusion |

---

## 💻 Sử dụng

### 1. Interactive Mode (chế độ tương tác)

```bash
python main.py
```

Gõ câu hỏi và nhận câu trả lời kèm trích dẫn pháp lý:

```
❓ Câu hỏi: Giáo dục chính quy là gì?
⏳ Đang xử lý...

📋 Kết quả
────────────────────────────────
Giáo dục chính quy là giáo dục theo khóa học trong cơ sở giáo dục
để thực hiện một chương trình giáo dục nhất định...

📌 Căn cứ pháp lý:
- Luật 43/2019/QH14, Điều 5, Khoản 1
- Luật 08/2012/QH13, Điều 4, Khoản 1
```

#### Các lệnh trong interactive mode:

| Lệnh | Mô tả |
|------|-------|
| `/debug` | Bật/tắt hiển thị chi tiết suy luận |
| `/stats` | Xem thống kê pipeline (số nodes, chunks, edges) |
| `/tree` | Xem cấu trúc cây phân cấp của các luật |
| `/benchmark` | Chạy benchmark 12 câu hỏi |
| `/quit` | Thoát chương trình |

### 2. Single Query Mode

```bash
python main.py -q "Điều kiện thành lập trường đại học?"
```

### 3. Debug Mode (hiển thị reasoning trace)

```bash
python main.py --debug
```

Hoặc trong interactive mode, gõ `/debug` để bật/tắt.

### 4. Xem cấu trúc cây

```bash
python main.py --tree
```

Output:
```
[root] Corpus Luật Giáo dục Việt Nam
├── [document] Luật 08/2012/QH13
│   ├── [chapter] Chương I
│   │   ├── [article] Điều 1. Phạm vi điều chỉnh
│   │   ├── [article] Điều 4. Giải thích từ ngữ
│   │   │   ├── [clause] Khoản 1
│   │   │   ├── [clause] Khoản 2
│   │   │   └── ...
│   ├── [chapter] Chương II
│   │   └── ...
├── [document] Luật 34/2018/QH14
│   └── ...
```

### 5. Xem thống kê

```bash
python main.py --stats
```

### 6. Sử dụng trong code Python

```python
from src.indexing import build_pipeline

# Build pipeline (chạy 1 lần, mất ~28s)
pipeline = build_pipeline()

# Truy vấn
result = pipeline.query("Giáo dục chính quy là gì?")
print(result)  # Markdown formatted answer

# Truy vấn với debug
result = pipeline.query("...", show_reasoning=True)

# Lấy raw object (cho xử lý programmatic)
answer = pipeline.query_raw("...")
print(answer.citations)
print(answer.confidence)
print(answer.reasoning_trace)
```

---

## 📊 Benchmark

### Chạy benchmark

```bash
python main.py --benchmark
```

### Kết quả benchmark (12 câu hỏi, 5 categories)

| Metric | Value |
|--------|-------|
| Avg Recall@K | **0.854** |
| Avg Keyword Coverage | **0.958** |
| Avg Precision@K | 0.686 |
| Intent Accuracy | 75.0% |

| Category | Count | Precision | Recall | Keywords | Intent |
|----------|-------|-----------|--------|----------|--------|
| definition | 4 | 0.223 | 0.688 | 1.000 | 100% |
| condition | 3 | 0.778 | 0.917 | 1.000 | 33% |
| comparison | 1 | 1.000 | 0.750 | 0.500 | 100% |
| general | 3 | 1.000 | 1.000 | 1.000 | 67% |
| procedure | 1 | 1.000 | 1.000 | 1.000 | 100% |

### Thêm câu hỏi benchmark tùy chỉnh

Sửa file `src/benchmark.py`, thêm vào `DEFAULT_BENCHMARK`:

```python
BenchmarkQuestion(
    question="Câu hỏi của bạn?",
    intent="definition",  # definition|condition|procedure|comparison|general
    expected_articles=["Điều X", "Điều Y"],
    expected_sources=["Luật 08/2012/QH13"],
    expected_keywords=["từ khóa 1", "từ khóa 2"],
    difficulty="medium",
    category="your_category",
)
```

---

## 🧠 Nâng cao: LLM Knowledge Graph Enhancement

Nếu có OpenAI API key, bạn có thể mở rộng Knowledge Graph bằng LLM:

```bash
# Mở rộng KG với 50 chunks (mặc định)
python main.py --enhance-kg

# Mở rộng với 200 chunks
python main.py --enhance-kg --enhance-kg-max 200
```

Tính năng này sử dụng LLM để trích xuất thêm:
- Entities chi tiết (concepts, organizations, roles, processes)
- Relations phức tạp (DEFINES, REGULATES, REQUIRES, GRANTS, PROHIBITS)
- Liên kết chéo giữa các văn bản luật

---

## 📁 Cấu trúc dự án

```
testBookRAG/
├── data/
│   └── final/                        # 8 file JSON luật giáo dục (input)
│       ├── Luật 08_2012_QH13.json
│       ├── Luật 34_2018_QH14.json
│       ├── Luật 43_2019_QH14.json
│       └── ...
├── indexes/                          # Benchmark reports (auto-generated)
├── src/
│   ├── __init__.py
│   ├── config.py                     # Cấu hình & hằng số
│   ├── models.py                     # 12 Pydantic data models
│   ├── pipeline.py                   # Pipeline orchestrator (retry loop)
│   ├── indexing.py                   # Build tất cả indexes
│   ├── benchmark.py                  # Framework benchmark (12 questions)
│   ├── bookindex/
│   │   ├── __init__.py
│   │   ├── tree_builder.py           # Cây phân cấp 6 tầng
│   │   ├── kg_builder.py             # Knowledge graph (rule-based)
│   │   └── kg_enhancer.py            # KG enhancement (LLM-assisted)
│   ├── query/
│   │   ├── __init__.py
│   │   └── processor.py              # Intent classification + NER
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py           # Dense retrieval (Qdrant in-memory)
│   │   ├── sparse_retrieval.py       # BM25 retrieval
│   │   ├── hybrid_retrieval.py       # Hybrid fusion (8 bước)
│   │   └── reranker.py              # Cross-encoder reranker
│   ├── reasoning/
│   │   ├── __init__.py
│   │   └── reasoning_engine.py       # CoT reasoning (4 bước)
│   └── generation/
│       ├── __init__.py
│       ├── answer_generator.py       # LLM answer + citations
│       └── formatter.py              # Markdown formatting
├── main.py                           # CLI entry point
├── requirements.txt                  # Python dependencies
├── .env.example                      # Template cho API keys
└── README.md                         # File này
```

---

## 🔧 Chi tiết kỹ thuật

### Hierarchical Tree

Cây phân cấp được xây dựng tự động từ metadata trong JSON:
- `metadata.source` → Document node
- `metadata.chapter` → Chapter node
- `metadata.section` → Section node (nếu có)
- `metadata.article_number` → Article node
- `metadata.clause_number` (khi `is_sub_split=true`) → Clause node
- `metadata.point_label` → Point node

### Hybrid Retrieval Pipeline (8 bước)

1. **Dense retrieval** – Semantic search qua Qdrant (cosine similarity)
2. **Sparse retrieval** – BM25 keyword search với Vietnamese tokenization
3. **KG retrieval** – Graph traversal từ concepts → related chunks
4. **RRF Fusion** – Reciprocal Rank Fusion kết hợp 3 nguồn
5. **Intent boosting** – Tăng score cho "Giải thích từ ngữ" khi hỏi định nghĩa
6. **Deduplication** – Loại chunks trùng (cùng điều/khoản/điểm)
7. **Cross-encoder reranking** – Rerank bằng cross-encoder hoặc heuristic
8. **Context expansion** – Đảm bảo đa nguồn cho câu hỏi so sánh

### Fallback Mode

Khi không có API key, pipeline hoạt động ở chế độ fallback:
- Query processing: ✅ đầy đủ (rule-based, không cần LLM)
- Retrieval: ✅ đầy đủ (dense + sparse + KG)
- Reasoning: ⚠️ basic analysis (không có LLM reasoning)
- Answer: ⚠️ trích dẫn trực tiếp từ luật (không qua LLM tổng hợp)

---

## ❓ FAQ

**Q: Lần đầu chạy rất lâu?**
A: Lần đầu cần tải embedding model (~470MB). Các lần sau sẽ nhanh hơn (~28s để build indexes).

**Q: Có thể dùng model embedding khác không?**
A: Có. Sửa `EMBEDDING_MODEL` trong `.env` và cập nhật `EMBEDDING_DIM` trong `src/config.py`.

**Q: Có thể dùng Gemini/Claude thay OpenAI không?**
A: Cần sửa `src/reasoning/reasoning_engine.py` và `src/generation/answer_generator.py` để dùng SDK tương ứng.

**Q: Dữ liệu được lưu ở đâu?**
A: Tất cả indexes chạy in-memory (RAM). Không persisted ra disk. Mỗi lần chạy sẽ rebuild.

**Q: Làm sao thêm luật mới?**
A: Thêm file JSON mới vào `data/final/` theo cùng format, pipeline sẽ tự động nhận.

---

## 📄 License

MIT License
