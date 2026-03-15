import os
import json
import glob  
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "legal_documents"

DATA_PATTERN = "data/final/*.json" 

client_q = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
model = SentenceTransformer("keepitreal/vietnamese-sbert")
VECTOR_SIZE = 768

def create_collection():
    try:
        client_q.get_collection(collection_name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' đã có sẵn.")
    except:
        print(f"Đang tạo collection mới '{COLLECTION_NAME}'...")
        client_q.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )

def upsert_json_data():
    files = glob.glob(DATA_PATTERN)
    
    if not files:
        print(f"Không tìm thấy file JSON nào tại {DATA_PATTERN}")
        
        print(f"Thư mục hiện tại: {os.getcwd()}")
        return

    all_points = []
    global_id = 0

    for file_path in files:
        print(f"Đang đọc file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        print(f"Đang tạo vector cho {len(data_list)} điều luật trong file này...")
        for item in data_list:
            text_to_embed = item['content']['full_text']
            
            payload = {
                "text": text_to_embed,
                "source": item['metadata']['source'],
                "chapter": item['metadata']['chapter'],
                "article_number": item['metadata']['article_number'],
                "article_title": item['metadata']['article_title'],
                "tags": item['metadata']['tags']
            }
            
            vector = model.encode(text_to_embed).tolist()
            
            all_points.append(models.PointStruct(
                id=global_id, 
                vector=vector,
                payload=payload
            ))
            global_id += 1

    # Bơm toàn bộ dữ liệu từ tất cả các file vào Qdrant
    if all_points:
        print(f"Đang bơm tổng cộng {len(all_points)} điểm dữ liệu vào Qdrant...")
        client_q.upsert(collection_name=COLLECTION_NAME, points=all_points)
        print("Đã nạp xong toàn bộ dữ liệu vào Qdrant!")

if __name__ == "__main__":
    create_collection()
    upsert_json_data()