import os
import json
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from groq import Groq

class VietnameseEmbedding:
    def __init__(self, model_name="keepitreal/vietnamese-sbert"):
        self.model = SentenceTransformer(model_name)
    def name(self): return "VietnameseEmbedding"
    def embed_query(self, input):
        if isinstance(input, str): input = [input]
        return self.model.encode(input).tolist()
    def __call__(self, input):
        if isinstance(input, str): input = [input]
        return self.model.encode(input).tolist()

# --- HÀM LỌC KNOWLEDGE GRAPH THÔNG MINH ---
def get_relevant_kg_triples(target_article_nums, kg_path):
    if not os.path.exists(kg_path):
        return ""
    try:
        with open(kg_path, 'r', encoding='utf-8') as f:
            kg_data = json.load(f)
        
        relevant_triples = []
        for item in kg_data:
            triple = item.get('triple', {})
            sub = str(triple.get('subject', '')).lower()
            obj = str(triple.get('object', '')).lower()
            
            # Lọc theo số điều vừa tìm thấy (ví dụ: '17', '18')
            if any(f"dieu_{num}" in sub or f"dieu_{num}" in obj for num in target_article_nums):
                info = f"- {triple['subject']} {triple['relation']} {triple['object']} (Trích dẫn: {item.get('quote', 'N/A')})"
                relevant_triples.append(info)
        
        return "\n".join(relevant_triples[:10])
    except:
        return ""

viet_em = VietnameseEmbedding()
# --- HÀM CHÍNH DÙNG QDRANT ---
def get_legal_answer(prompt, collection, client_groq, kg_path):
    try:
        # 1. Kết nối Qdrant
        q_host = os.getenv("QDRANT_HOST", "qdrant")  # Đổi mặc định từ localhost sang qdrant
        q_port = int(os.getenv("QDRANT_PORT", 6333))
        client_q = QdrantClient(host=q_host, port=q_port)
        
        # Khởi tạo embedding để query
        query_vector = viet_em.embed_query(prompt)[0]

        # 2. VECTOR SEARCH TRÊN QDRANT
        search_results = client_q.query_points(
            collection_name="legal_documents",
            query=query_vector, 
            limit=10
        ).points # .points ở cuối để lấy danh sách kết quả

        vector_docs = [hit.payload.get('text', '') for hit in search_results]
        metas = [hit.payload for hit in search_results]
        found_article_nums = [str(m.get('article_number', '')) for m in metas]

        # 3. BM25 SEARCH (Lấy toàn bộ docs từ Qdrant để làm corpus)
        all_points = client_q.scroll(collection_name="legal_documents", limit=100)[0]
        all_docs = [p.payload.get('text', '') for p in all_points]
        
        if all_docs:
            tokenized_corpus = [doc.lower().split() for doc in all_docs]
            bm25 = BM25Okapi(tokenized_corpus)
            bm25_docs = bm25.get_top_n(prompt.lower().split(), all_docs, n=3)
        else:
            bm25_docs = []

        # 4. HYBRID & KG FILTERING
        combined_docs = list(dict.fromkeys(bm25_docs + vector_docs))
        context_rag = "\n\n".join(combined_docs[:5])
        kg_info = get_relevant_kg_triples(found_article_nums, kg_path)

        # 5. GỌI AI (GROQ)
        ai_prompt = f"""
        Mày là một Luật sư chuyên gia về hệ thống Pháp luật Việt Nam. 
        Nhiệm vụ của mày là trả lời câu hỏi của khách hàng một cách CHI TIẾT, ĐẦY ĐỦ và TRANG TRỌNG dựa trên dữ liệu cung cấp.

        YÊU CẦU TRẢ LỜI:
        1. Phải trích dẫn cụ thể các Điều, Khoản (nếu có trong dữ liệu).
        2. Phải giải thích rõ từng ý: Đối tượng là ai? Điều kiện là gì? Chính sách cụ thể ra sao?
        3. Nếu dữ liệu có nhiều văn bản, hãy tổng hợp lại thành một câu trả lời toàn diện, không bỏ sót thông tin nào.
        4. Trình bày Markdown rõ ràng với các tiêu đề (##), danh sách gạch đầu dòng và in đậm các từ khóa quan trọng.
        5. CHỈ nhắc đến trạng thái hiệu lực NẾU thông tin QUAN HỆ (KG) có chứa từ khóa 'BI_BAI_BO_BOI'. 
        6. TUYỆT ĐỐI KHÔNG giải thích về hệ thống, không nhắc đến các từ khóa kỹ thuật như 'KG', 'BI_BAI_BO_BOI' hay 'Văn bản cung cấp' nếu dữ liệu đó không tồn tại hoặc không liên quan.

        VĂN BẢN: {context_rag}
        QUAN HỆ (KG): {kg_info}
        CÂU HỎI: {prompt}
        """

        completion = client_groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": ai_prompt}],
            temperature=0.4,
            max_tokens=4096
        )

        return completion.choices[0].message.content, context_rag, metas

    except Exception as e:
        return f"Lỗi hệ thống core_logic (Qdrant): {str(e)}", "", []