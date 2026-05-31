# LawEdu AI - Legal Education Chatbot ⚖️🎓

LawEdu AI is an advanced Legal Education Chatbot built with a **RAG (Retrieval-Augmented Generation) Pipeline** and an **Agentic AI Workflow**. The application leverages powerful LLMs to answer legal queries efficiently, backed by a vector and knowledge graph hybrid search.

## ✨ Features

- **Hybrid Search RAG:** Combines Knowledge Graph, FAISS Vector Search, and BM25 to deliver highly accurate context.
- **Dual LLM Architecture:** Uses a fast 20B model for intent classification & query rewriting, and a powerful 120B model for complex legal text generation.
- **Modern User Interface:** A stunning, animated UI built with React, Vite, Tailwind CSS, and Shadcn UI.
- **FastAPI Backend:** High-performance async API with built-in rate-limiting and auth middleware.
- **Caching Layer:** Redis-powered caching for faster repeated queries.

## 🛠 Tech Stack

### Backend
- **Framework:** FastAPI, Uvicorn
- **AI/LLM:** OpenAI-compatible API
- **RAG:** Sentence-Transformers, FAISS, NetworkX, Rank-BM25
- **Data:** Redis, PyYAML, Python-Dotenv

### Frontend
- **Framework:** React 19, TypeScript, Vite
- **Styling:** Tailwind CSS v4, Framer Motion, clsx, tailwind-merge
- **Icons:** Lucide React

## 📂 Project Structure

```text
legal-edu-chatbot/
├── app/               # FastAPI Backend Source Code
│   ├── index/         # Graph & Vector Indexing
│   ├── llm/           # LLM Clients & Prompts
│   ├── query/         # Query Expansion & Intent Classification
│   ├── rag/           # Hybrid Search & Context Building
│   └── main.py        # FastAPI Application Entry
├── frontend/          # React + Vite Frontend
│   ├── src/           # React Components & Views
│   └── package.json   # Node Dependencies
├── data/              # Legal documents and parsed chunks
├── outputs/           # Output graphs and evaluation results
├── requirements.txt   # Python Dependencies
└── .env               # Environment Variables Configuration
```

## ⚙️ System Pipeline Flow

The system orchestrates an advanced retrieval-augmented generation pipeline using dual LLMs:

```mermaid
flowchart TD
    Client[👤 User Client] -->|Sends Legal Query| API[🛡️ API Gateway]
    API -->|Auth & Rate Limit| Router[🚦 Chat Orchestrator]
    Router -->|Check Redis| Cache{📦 Cache Hit?}
    Cache -->|Yes| ReturnCache[✅ Return Cached Response]
    Cache -->|No| Rewrite[🧠 Query Intent & Rewrite<br/>(Agentic 20B LLM)]
    Rewrite --> Hybrid[🔍 Hybrid Search RAG]
    Hybrid -->|Vector, BM25, Knowledge Graph| ReRank[📊 Re-Ranker]
    ReRank --> Context[📑 Context Builder]
    Context --> Generator[🤖 LLM Generator<br/>(Primary 120B LLM)]
    Generator -->|Generate Legal Answer| SaveCache[💾 Update Cache]
    SaveCache -->|Response| Client
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/legal-edu-chatbot.git
cd legal-edu-chatbot
```

### 2. Set up the Backend

Make sure you have Python 3.9+ installed.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables

Create or configure the `.env` file in the root directory based on the following template (you can copy `.env.example` if available):

```env
# Primary LLM API (Generation)
LLM_API_BASE=https://api.groq.com/openai/v1
LLM_API_KEY=your_api_key_here
LLM_MODEL_NAME=openai/gpt-oss-120b

# Secondary LLM API (Agentic Tasks)
GEN_LLM_API_BASE=https://api.groq.com/openai/v1
GEN_LLM_API_KEY=your_api_key_here
GEN_LLM_MODEL_NAME=openai/gpt-oss-20b

# Redis Cache (Optional)
REDIS_URL=redis://localhost:6379/0

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=lawedu-default-key
```

### 4. Run the Backend Server

```bash
# From the project root (legal-edu-chatbot folder)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
The API will be available at `http://localhost:8000`. You can view the automatic interactive API documentation at `http://localhost:8000/docs`.

### 5. Set up and Run the Frontend

Open a new terminal window.

```bash
cd frontend

# Install Node dependencies
npm install

# Start the development server
npm run dev
```

The frontend will start on `http://localhost:5173` (or another port specified by Vite). Open this URL in your browser to interact with LawEdu AI!

## 📝 License
This project is for educational and research purposes.