import json
import re

INPUT_FILE = r"C:\legal-edu-chatbot\data\Nghi dinh\04_2021_NĐ_CP.json"
OUTPUT_FILE = r"C:\legal-edu-chatbot\data\Nghi dinh\04_2021_NĐ_CP_chunked.json"

# ==========================================
# CÁCH 1: HIERARCHICAL CHUNKING (LỒNG NHAU)
# ==========================================

def parse_points(text):
    """Parse điểm (a, b, c...) từ text của khoản"""
    # Pattern: bắt đầu bằng a), b), c)... ở đầu dòng hoặc sau khoảng trắng
    point_pattern = re.compile(r'(?:^|\n)([a-zđ]\))\s')
    matches = list(point_pattern.finditer(text))
    
    if not matches:
        return []
    
    points = []
    for i, match in enumerate(matches):
        label = match.group(1).rstrip(')')  # "a", "b", "c"...
        start = match.start()
        # Bỏ ký tự newline đầu nếu có
        if text[start] == '\n':
            start += 1
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        # Bỏ newline đầu đoạn tiếp theo
        point_text = text[start:end].strip()
        points.append({
            "label": label,
            "text": point_text
        })
    return points


def parse_clauses(full_text, article_title):
    """Parse khoản (1, 2, 3...) từ full_text của điều"""
    # Bỏ dòng tiêu đề điều ở đầu
    lines = full_text.strip().split('\n')
    body = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ''

    if not body:
        return []

    # Pattern khoản: số ở đầu dòng theo sau là dấu chấm và khoảng trắng
    clause_pattern = re.compile(r'(?:^|\n)(\d+)\.\s')
    matches = list(clause_pattern.finditer(body))

    if not matches:
        return []

    clauses = []
    for i, match in enumerate(matches):
        clause_id = match.group(1)
        start = match.start()
        if body[start] == '\n':
            start += 1
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        clause_text = body[start:end].strip()

        points = parse_points(clause_text)

        clause_item = {
            "clause_id": clause_id,
            "text": clause_text
        }
        
        # Chỉ thêm thuộc tính point nếu mảng points không rỗng
        if points:
            clause_item["point"] = points
            
        clauses.append(clause_item)
    return clauses


def chunk_articles(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = []

    for article in data:
        full_text = article.get('content', {}).get('full_text', '')
        article_title = article.get('metadata', {}).get('article_title', '')

        clauses = parse_clauses(full_text, article_title)
        
        content_dict = {
            "full_text": full_text
        }
        
        # Chỉ thêm thuộc tính clause nếu mảng clauses không rỗng
        if clauses:
            content_dict["clause"] = clauses

        new_article = {
            "id": article.get("id"),
            "metadata": article.get("metadata"),
            "content": content_dict
        }
        result.append(new_article)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ Hierarchical chunking hoàn thành! Đã xử lý {len(result)} articles -> {output_path}")

    # In thống kê (kiểm tra an toàn vì 'clause'/'point' có thể không tồn tại nữa)
    total_clauses = sum(len(a['content'].get('clause', [])) for a in result)
    total_points = sum(
        len(c.get('point', []))
        for a in result
        for c in a['content'].get('clause', [])
    )
    articles_no_clause = sum(1 for a in result if not a['content'].get('clause'))
    print(f"  Articles: {len(result)}")
    print(f"  Articles không có khoản (điều đơn): {articles_no_clause}")
    print(f"  Tổng khoản: {total_clauses}")
    print(f"  Tổng điểm: {total_points}")


# ==========================================
# CÁCH 2: FLAT CHUNKING (CHIA NHỎ TỪNG CẤP)
# ==========================================

def parse_diem_flat(text):
    """
    Tách các điểm (a), b), c)...) trong một khoản.
    Trả về list: [{"diem": "a", "text": "..."}, ...]
    """
    pattern = r'(?:^|\n)([a-zđ]{1,2})\)\s'
    parts = re.split(pattern, text)

    if len(parts) == 1:
        return []

    diem_list = []
    i = 1
    while i < len(parts) - 1:
        label = parts[i]
        content = parts[i + 1].strip()
        diem_list.append({"diem": label, "text": f"{label}) {content}"})
        i += 2

    return diem_list


def parse_khoan_flat(full_text, article_title):
    """
    Tách các khoản (1., 2., 3.,...) trong một điều.
    Trả về list: [{"khoan": "1", "title": "...", "text": "...", "diem": [...]}, ...]
    """
    # Bỏ dòng tiêu đề điều ở đầu
    lines = full_text.split('\n')
    body = '\n'.join(lines[1:]).strip()  # bỏ dòng "Điều X. ..."

    # Pattern khoản: dòng bắt đầu bằng số và dấu chấm (1. 2. 3. ...)
    pattern = r'(?:^|\n)(\d+)\.\s'
    parts = re.split(pattern, body)

    if len(parts) == 1:
        # Không có khoản → toàn bộ nội dung là 1 đoạn
        return []

    khoan_list = []
    i = 1
    while i < len(parts) - 1:
        khoan_num = parts[i]
        khoan_text_raw = parts[i + 1].strip()

        diem_list = parse_diem_flat(khoan_text_raw)

        khoan_list.append({
            "khoan": khoan_num,
            "text": f"{khoan_num}. {khoan_text_raw}",
            "diem": diem_list
        })
        i += 2

    return khoan_list


def chunk_article_flat(article):
    """
    Từ 1 article gốc, sinh ra list các chunk nhỏ hơn (cấp Điều, Khoản, Điểm).
    """
    chunks = []
    meta = article["metadata"]
    full_text = article["content"]["full_text"]
    art_id = article["id"]

    # --- Chunk cấp Điều (giữ nguyên) ---
    chunks.append({
        "id": art_id,
        "level": "dieu",
        "metadata": {
            **meta,
            "khoan_number": None,
            "diem_label": None,
            "parent_id": None,
        },
        "content": {
            "full_text": full_text
        }
    })

    # --- Tách khoản ---
    khoan_list = parse_khoan_flat(full_text, meta.get("article_title", ""))

    for khoan in khoan_list:
        khoan_id = f"{art_id}_K{khoan['khoan']}"

        # Chunk cấp Khoản
        chunks.append({
            "id": khoan_id,
            "level": "khoan",
            "metadata": {
                **meta,
                "khoan_number": khoan["khoan"],
                "diem_label": None,
                "parent_id": art_id,
            },
            "content": {
                "full_text": khoan["text"]
            }
        })

        # --- Tách điểm trong khoản ---
        for diem in khoan["diem"]:
            diem_id = f"{khoan_id}_D{diem['diem']}"

            chunks.append({
                "id": diem_id,
                "level": "diem",
                "metadata": {
                    **meta,
                    "khoan_number": khoan["khoan"],
                    "diem_label": diem["diem"],
                    "parent_id": khoan_id,
                },
                "content": {
                    "full_text": diem["text"]
                }
            })

    return chunks


def main_flat(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_chunks = []
    stats = {"dieu": 0, "khoan": 0, "diem": 0}

    for article in data:
        chunks = chunk_article_flat(article)
        all_chunks.extend(chunks)
        for c in chunks:
            stats[c["level"]] += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ Flat chunking hoàn thành! Tổng chunks: {len(all_chunks)}")
    print(f"   - Cấp Điều  : {stats['dieu']}")
    print(f"   - Cấp Khoản : {stats['khoan']}")
    print(f"   - Cấp Điểm  : {stats['diem']}")
    print(f"📁 Đã lưu: {output_path}")

    # In mẫu vài chunk để kiểm tra
    print("\n--- MẪU CHUNK (Điều 5) ---")
    samples = [c for c in all_chunks if "D5" in c["id"]]
    for s in samples[:6]:
        print(f"\n[{s['level'].upper()}] id={s['id']}")
        print(f"  text: {s['content']['full_text'][:120]}...")


if __name__ == '__main__':
    # Chạy Cách 1 (Hierarchical Chunking) - Theo yêu cầu mới nhất của User
    chunk_articles(INPUT_FILE, OUTPUT_FILE)
    
    # Bỏ comment dòng dưới nếu muốn chạy Cách 2 (Flat Chunking)
    # main_flat(INPUT_FILE, OUTPUT_FILE.replace(".json", "_flat.json"))