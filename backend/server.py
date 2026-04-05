from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
from neo4j import GraphDatabase
import os
from core_logic import get_legal_answer

# 1. Khởi tạo FastAPI
app = FastAPI(title="Legal AI API Server (Hybrid Graph + Vector)", version="2.0")

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "password123"

# Khởi tạo Groq Client
client_groq = Groq(api_key=GROQ_API_KEY)

neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

class QuestionRequest(BaseModel):
    prompt: str
# --- Hàm lấy context từ Knowledge Graph ---
def get_kg_context(query_text):
    """
    Hàm này sẽ lục trong Neo4j xem có thực thể nào (DoiTuong, CapHoc...) 
    khớp với câu hỏi không, rồi lấy các Điều luật liên quan.
    """
    with neo4j_driver.session() as session:
        # Query Cypher tìm các Điều luật nối với thực thể xuất hiện trong query
        cypher = """
        MATCH (e) 
        WHERE toLower($query) CONTAINS toLower(e.ten)
        MATCH (d:Dieu)-[:LIÊN_QUAN_ĐẾN]->(e)
        OPTIONAL MATCH (d)-[:THAM_CHIEU]->(d_ref:Dieu)
        RETURN d.ten_dieu as title, d.noi_dung as content, collect(d_ref.ten_dieu) as refs
        LIMIT 3
        """
        results = session.run(cypher, query=query_text)
        kg_context = ""
        for record in results:
            kg_context += f"\n- {record['title']}: {record['content']}"
            if record['refs']:
                kg_context += f" (Dẫn chiếu nội bộ: {', '.join(record['refs'])})"
        return kg_context
    
# 2. Định nghĩa Endpoint chính
@app.post("/ask")
async def ask_legal_bot(request: QuestionRequest):
    try:
        # A. Lấy context từ Qdrant (Vector Search) thông qua core_logic
        answer, vector_context, metadatas = get_legal_answer(
            request.prompt, 
            None, 
            client_groq, 
            "outputs/knowledge_graph/legal_graph_final.json"
        )

        # B. Lấy thêm context từ Neo4j (Graph Search) - MỚI
        kg_context = get_kg_context(request.prompt)
        
        # C. Nếu có dữ liệu từ Graph, ta "nhồi" thêm vào câu trả lời hoặc 
        # Cập nhật lại logic gọi Groq ở đây (Hoặc sửa trong core_logic.py)
        # Ở đây tao giả định mày muốn gộp context để AI trả lời thông minh hơn
        if kg_context:
            new_prompt = f"Dựa trên các mối liên hệ thực thể sau: {kg_context}\n\nVà nội dung chi tiết: {vector_context}\n\nHãy trả lời: {request.prompt}"
            # Gọi Groq lại một lần nữa hoặc cập nhật lại answer
        
        return {
            "status": "success",
            "answer": answer,
            "sources": metadatas,
            "raw_context": vector_context,
            "graph_context": kg_context
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