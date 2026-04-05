import json
import os
import time
from groq import Groq
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- CẤU HÌNH ĐƯỜNG DẪN ---
INPUT_FILE = r'D:/demo/outputs/knowledge_graph/legal_graph_skeleton.json'
FINAL_FILE = r'D:/demo/outputs/knowledge_graph/legal_graph_final.json'

def ask_ai_extract_entities(text, doc_name):
    """
    Dùng AI nhặt thực thể mềm: Đối tượng, Cấp học, Khái niệm chuyên ngành.
    """
    prompt = f"""
    Mày là chuyên gia phân tích Luật Giáo dục Việt Nam. 
    Từ đoạn văn bản của '{doc_name}', hãy trích xuất các thực thể quan trọng sau:
    1. DoiTuong: Người hoặc tổ chức (VD: Nhà đầu tư, Giáo viên, Học sinh, Tổ chức kinh tế).
    2. CapHoc: Cấp bậc hoặc trình độ (VD: Mầm non, Tiểu học, Trung cấp, Đại học).
    3. KhaiNiem: Các thuật ngữ chuyên môn (VD: Đào tạo chính quy, Tín chỉ, Mô-đun, Tư thục).

    VĂN BẢN: {text}
    
    YÊU CẦU:
    - Trả về DUY NHẤT một JSON object có định dạng: {{"entities": [{{"name": "...", "type": "DoiTuong/CapHoc/KhaiNiem"}}]}}
    - Không giải thích gì thêm.
    """
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        time.sleep(0.8) 
        
        res_content = response.choices[0].message.content
        return json.loads(res_content).get("entities", [])
    except Exception as e:
        print(f"\n❌ Lỗi API tại đoạn văn: {text[:50]}... | Lỗi: {e}")
        if "429" in str(e):
            print("🛑 Chạm trần Rate Limit rồi! Nghỉ 60s...")
            time.sleep(60)
        return []

# 1. Đọc dữ liệu từ Step 1 (hoặc file Final nếu chạy dở)
if os.path.exists(FINAL_FILE):
    print("🔄 Tìm thấy file chạy dở, tiếp tục từ checkpoint...")
    with open(FINAL_FILE, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
else:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)

# 2. Bắt đầu quy trình "đắp thịt"
print(f"🚀 Bắt đầu dùng AI xử lý {len(graph_data)} điều luật...")

count = 0
try:
    for item in tqdm(graph_data):
        # Chỉ chạy những cái chưa có entities (Cơ chế Checkpoint)
        if not item.get("entities"):
            item["entities"] = ask_ai_extract_entities(item["content"], item["source"])
            count += 1
            
            # Cứ 10 câu thì lưu file một lần cho chắc ăn
            if count % 10 == 0:
                with open(FINAL_FILE, 'w', encoding='utf-8') as f:
                    json.dump(graph_data, f, ensure_ascii=False, indent=4)
except KeyboardInterrupt:
    print("\nDừng khẩn cấp! Đang lưu dữ liệu đã xử lý...")

# 3. Lưu file kết quả cuối cùng
with open(FINAL_FILE, 'w', encoding='utf-8') as f:
    json.dump(graph_data, f, ensure_ascii=False, indent=4)

print(f"\n✅ HOÀN TẤT! Đã xử lý thêm {count} điều luật.")
print(f"📍 File cuối cùng nằm tại: {FINAL_FILE}")