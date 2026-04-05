import os
import json
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from groq import Groq
from neo4j import GraphDatabase

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

# --- HÀM TRUY VẤN NEO4J  ---
def get_neo4j_context(target_article_nums, source_law):
    uri = "bolt://neo4j:7687"
    user = "neo4j"
    password = "password123" 
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    context_list = []
    
        # Chuyển tất cả số điều thành string để so sánh trong Cypher
    nums = [str(n) for n in target_article_nums]
    
    with driver.session() as session:
        cypher = """
        MATCH (d:Dieu)
        WHERE d.so_dieu IN $nums AND d.id CONTAINS $law_prefix
        OPTIONAL MATCH (d)-[:LIÊN_QUAN_ĐẾN]->(ent)
        OPTIONAL MATCH (d)-[:THAM_CHIEU]->(d_ref:Dieu)
        RETURN d.so_dieu as so, 
               collect(DISTINCT ent.ten) as entities, 
               collect(DISTINCT d_ref.so_dieu) as refs
        """
        law_prefix = source_law.split('/')[0].replace(" ", "") 
        results = session.run(cypher, nums=nums, law_prefix=law_prefix)
        
        for record in results:
            entities_str = ", ".join(record['entities']) if record['entities'] else "Không có thực thể"
            info = f"Điều {record['so']} liên quan đến: {entities_str}"
            
            if record['refs']:
                info += f" [REF] {', '.join(record['refs'])}"
            
            context_list.append(info)
            
    driver.close()
    return "\n".join(context_list)

viet_em = VietnameseEmbedding()

def get_legal_answer(prompt, collection, client_groq, kg_path):
    try:
        # 1. Kết nối Qdrant
        q_host = os.getenv("QDRANT_HOST", "localhost") 
        q_port = int(os.getenv("QDRANT_PORT", 6333))
        client_q = QdrantClient(host=q_host, port=q_port)
        
        query_vector = viet_em.embed_query(prompt)[0]

        # 2. VECTOR SEARCH
        search_results = client_q.query_points(
            collection_name="legal_documents",
            query=query_vector, 
            limit=5
        ).points 

        vector_docs = [hit.payload.get('text', '') for hit in search_results]
        metas = [hit.payload for hit in search_results]
        
        # Lấy danh sách số điều và tên luật để hỏi Neo4j
        found_article_nums = [str(m.get('article_number', '')) for m in metas]
        source_law = metas[0].get('source', '') if metas else ""

        # 3. BM25 SEARCH (Hybrid)
        all_points = client_q.scroll(collection_name="legal_documents", limit=100)[0]
        all_docs = [p.payload.get('text', '') for p in all_points]
        
        bm25_docs = []
        if all_docs:
            tokenized_corpus = [doc.lower().split() for doc in all_docs]
            bm25 = BM25Okapi(tokenized_corpus)
            bm25_docs = bm25.get_top_n(prompt.lower().split(), all_docs, n=2)

        # 4. LẤY CONTEXT TỪ NEO4J
        context_rag = "\n\n".join(list(dict.fromkeys(bm25_docs + vector_docs)))
        kg_info = get_neo4j_context(found_article_nums, source_law)

        # 5. GỌI AI (GROQ)
        ai_prompt = f"""
        Mày là Luật sư chuyên gia Pháp luật Việt Nam. 
        Hãy trả lời câu hỏi dựa trên VĂN BẢN và QUAN HỆ (GRAPH) được cung cấp.

        Yêu cầu:
        - Ưu tiên thông tin từ VĂN BẢN và GRAPH để trích dẫn Điều luật cụ thể.
        - Nếu GRAPH có các thực thể liên quan, hãy lồng ghép vào câu trả lời để làm rõ đối tượng.
        - Nếu có nhiều phiên bản của cùng một Điều luật, hãy tổng hợp và giải thích sự khác biệt nếu có.
        - Trích dẫn đúng con số từ Điều luật và tên văn bản (nếu có) để người đọc dễ dàng tra cứu.
        - Giải thích dễ hiểu, chuyên nghiệp theo phong cách luật sư.

        VĂN BẢN: {context_rag}
        QUAN HỆ (GRAPH): {kg_info}
        CÂU HỎI: {prompt}
        """

        completion = client_groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": ai_prompt}],
            temperature=0.3
        )

        return completion.choices[0].message.content, context_rag, metas, kg_info

    except Exception as e:
        return f"Lỗi hệ thống core_logic: {str(e)}", "", []