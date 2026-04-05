# step1_build_skeleton.py
import json
import os
import glob
import re
from tqdm import tqdm

# Cấu hình đường dẫn của mày
DATA_DIR = r'D:/demo/data/final'
OUTPUT_FILE = r'D:/demo/outputs/knowledge_graph/legal_graph_skeleton.json'

def extract_references(text, current_article_number):
    """
    Nâng cấp Regex để nhặt Law Refs và tránh tự dẫn chiếu chính mình
    """
    # 1. Nhặt số điều: Điều 12, Điều 15...
    dieu_refs = re.findall(r"Điều\s+(\d+)", text)
    # Loại bỏ chính số điều hiện tại để tránh loop
    dieu_refs = [r for r in dieu_refs if r != str(current_article_number)]
    
    # 2. Nhặt tên các Luật khác (Tìm chữ 'Luật' đi kèm với các từ viết hoa phía sau)
    # Ví dụ: "Luật giáo dục", "Luật đầu tư"
    law_pattern = r"(Luật\s+[A-ZĐ][a-zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ\s]+)"
    law_refs = re.findall(law_pattern, text)
    
    # Làm sạch danh sách law_refs
    cleaned_law_refs = []
    for l in law_refs:
        clean_name = l.strip()
        # Chỉ lấy nếu có ít nhất 2 từ trở lên và không phải là "Luật này"
        if len(clean_name.split()) > 1 and "Luật này" not in clean_name:
            cleaned_law_refs.append(clean_name)

    return {
        "article_refs": list(set(dieu_refs)),
        "other_law_refs": list(set(cleaned_law_refs))
    }

skeleton_data = []
json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))

print(f"🚀 Đang xây khung xương từ {len(json_files)} file...")

for file_path in json_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
        for item in articles:
            text = item.get("content", {}).get("full_text", "")
            meta = item.get("metadata", {})
            
            # Lấy số điều hiện tại để truyền vào hàm Regex
            current_art_num = meta.get("article_number") 
            
            # --- ĐOẠN SỬA ĐÂY MÀY ƠI ---
            node = {
                "id": item.get("id"),
                "source": meta.get("source"),
                "chapter": meta.get("chapter"),
                "article_number": current_art_num,
                "title": meta.get("article_title"),
                "content": text,
                # Truyền thêm current_art_num vào hàm để nó biết đường mà tránh tự dẫn chiếu
                "refs": extract_references(text, current_art_num), 
                "entities": [] 
            }
            skeleton_data.append(node)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(skeleton_data, f, ensure_ascii=False, indent=4)

print(f"✅ Đã xong khung xương! Tổng cộng {len(skeleton_data)} điều luật.")