import json
import re
import os
import shutil

# ==========================================
# HIERARCHICAL CHUNKING
# ==========================================

def parse_points(text):
    point_pattern = re.compile(r'(?:^|\n)([a-zđ]\))\s')
    matches = list(point_pattern.finditer(text))

    if not matches:
        return []

    points = []
    for i, match in enumerate(matches):
        label = match.group(1).rstrip(')')
        start = match.start()

        if text[start] == '\n':
            start += 1

        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        point_text = text[start:end].strip()

        points.append({
            "label": label,
            "text": point_text
        })

    return points


def parse_clauses(full_text):
    lines = full_text.strip().split('\n')
    body = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ''

    if not body:
        return []

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

        if points:
            clause_item["point"] = points

        clauses.append(clause_item)

    return clauses


def chunk_articles_inplace(file_path):
    # Backup trước khi ghi đè
    backup_path = file_path + ".bak"
    shutil.copy(file_path, backup_path)

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = []

    for article in data:
        full_text = article.get('content', {}).get('full_text', '')

        clauses = parse_clauses(full_text)

        content_dict = {
            "full_text": full_text
        }

        if clauses:
            content_dict["clause"] = clauses

        new_article = {
            "id": article.get("id"),
            "metadata": article.get("metadata"),
            "content": content_dict
        }

        result.append(new_article)

    # Ghi đè file gốc
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Thống kê
    total_clauses = sum(len(a['content'].get('clause', [])) for a in result)
    total_points = sum(
        len(c.get('point', []))
        for a in result
        for c in a['content'].get('clause', [])
    )
    articles_no_clause = sum(1 for a in result if not a['content'].get('clause'))

    print(f"✅ Done: {os.path.basename(file_path)}")
    print(f"   Articles: {len(result)}")
    print(f"   Articles không có khoản: {articles_no_clause}")
    print(f"   Tổng khoản: {total_clauses}")
    print(f"   Tổng điểm: {total_points}")


# ==========================================
# PROCESS FOLDER
# ==========================================

def process_folder(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith(".json")]

    print(f"📂 Tìm thấy {len(files)} file JSON\n")

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        print(f"🚀 Đang xử lý: {file_name}")

        try:
            chunk_articles_inplace(file_path)
        except Exception as e:
            print(f"❌ Lỗi file {file_name}: {e}")


# ==========================================
# MAIN
# ==========================================

if __name__ == '__main__':
    FOLDER_PATH = r"C:\legal-edu-chatbot\data\Thong tu"
    process_folder(FOLDER_PATH)