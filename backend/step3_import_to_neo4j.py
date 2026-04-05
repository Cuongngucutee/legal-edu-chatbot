import json
from neo4j import GraphDatabase
from tqdm import tqdm

# --- CẤU HÌNH KẾT NỐI (Khớp với lệnh Docker mày vừa chạy) ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "password123" 
INPUT_FILE = r'D:/demo/outputs/knowledge_graph/legal_graph_final.json'

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def import_data(tx, item):
    # 1. Tạo Node Văn bản (Document) và Điều luật (Dieu)
    # Dùng MERGE để nếu chạy lại nó không tạo trùng Node
    query_base = """
    MERGE (doc:Document {id: $source})
    SET doc.title = $source
    
    MERGE (d:Dieu {id: $id})
    SET d.so_dieu = $so_dieu, 
        d.ten_dieu = $title, 
        d.noi_dung = $content
    
    MERGE (d)-[:THUOC_VAN_BAN]->(doc)
    """
    tx.run(query_base, 
           source=item['source'], 
           id=item['id'], 
           so_dieu=item['article_number'], 
           title=item['title'], 
           content=item['content'])

    # 2. Tạo quan hệ THAM_CHIEU giữa các Điều (Dẫn chiếu nội bộ)
    # Ví dụ: Điều 1 dẫn chiếu đến Điều 54
    for ref_so in item.get('refs', {}).get('article_refs', []):
        query_ref = """
        MATCH (d1:Dieu {id: $id})
        MATCH (d2:Dieu {so_dieu: $ref_so, source: $source})
        WHERE d1.id <> d2.id
        MERGE (d1)-[:THAM_CHIEU]->(d2)
        """
        tx.run(query_ref, id=item['id'], ref_so=str(ref_so), source=item['source'])

    # 3. Tạo quan hệ DẪN_CHIẾU_ĐẾN các Luật khác
    # Ví dụ: Luật Giáo dục dẫn chiếu sang Luật Doanh nghiệp
    for other_law in item.get('refs', {}).get('other_law_refs', []):
        query_other = """
        MATCH (d:Dieu {id: $id})
        MERGE (other:Document {id: $other_law})
        MERGE (d)-[:DẪN_CHIẾU_ĐẾN]->(other)
        """
        tx.run(query_other, id=item['id'], other_law=other_law)

    # 4. Tạo các thực thể mềm từ AI (DoiTuong, CapHoc, KhaiNiem)
    for ent in item.get('entities', []):
        label = ent['type'] # DoiTuong / CapHoc / KhaiNiem
        # Lưu ý: Cypher không cho truyền Label qua tham số trực tiếp, nên dùng f-string cho Label
        query_ent = f"""
        MATCH (d:Dieu {{id: $id}})
        MERGE (e:{label} {{ten: $name}})
        MERGE (d)-[:LIÊN_QUAN_ĐẾN]->(e)
        """
        tx.run(query_ent, id=item['id'], name=ent['name'])

# --- THỰC THI ---
try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"🚀 Đang nạp {len(data)} điều luật vào Neo4j...")
    with driver.session() as session:
        for item in tqdm(data):
            session.execute_write(import_data, item)
    
    print("\n✅ CHÚC MỪNG MÀY! Dữ liệu đã nằm gọn trong Neo4j.")
    print("👉 Giờ mở trình duyệt vào http://localhost:7474 gõ 'MATCH (n) RETURN n' để xem hàng!")

except Exception as e:
    print(f"\n❌ Lỗi rồi mày ơi: {e}")

finally:
    driver.close()