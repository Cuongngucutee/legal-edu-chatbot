from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import os
from core_logic import get_legal_answer

# 1. Khởi tạo FastAPI
app = FastAPI(title="Legal AI API Server (Qdrant Version)", version="1.0")

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
KG_PATH = "outputs/knowledge_graph/knowledge_graph_full.json"

# Khởi tạo Groq Client
client_groq = Groq(api_key=GROQ_API_KEY)

class QuestionRequest(BaseModel):
    prompt: str

# 2. Định nghĩa Endpoint chính
@app.post("/ask")
async def ask_legal_bot(request: QuestionRequest):
    try:
        # Gọi lại hàm logic từ file core_logic.py
        # CHÚ Ý: Tham số thứ 2 truyền là None vì core_logic sẽ tự connect Qdrant
        answer, context, metadatas = get_legal_answer(
            request.prompt, 
            None, 
            client_groq, 
            KG_PATH
        )
        
        return {
            "status": "success",
            "answer": answer,
            "sources": metadatas,
            "raw_context": context
        }
    except Exception as e:
        print(f"Lỗi Server: {e}") # Log lỗi ra terminal để dễ debug
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint kiểm tra trạng thái server
@app.get("/health")
async def health_check():
    # Thay collection.count() bằng thông báo đơn giản
    return {"status": "online", "database": "qdrant_mode"}

if __name__ == "__main__":
    import uvicorn
    # Để host 0.0.0.0 để Docker có thể ánh xạ cổng ra ngoài
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)