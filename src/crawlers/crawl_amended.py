import json
import os
import re
import unicodedata

class SuperLawSplitter:
    def __init__(self, input_folder='legal_data_json', output_folder='final_refined_data'):
        self.input_folder = input_folder
        self.output_folder = output_folder
        # Regex này sẽ tìm các dòng bắt đầu bằng số (1., 2...) và có chữ Điều ở sau
        self.re_split_point = re.compile(r'^(\d+)\.\s*(?:Sửa đổi|Bổ sung|Bãi bỏ).*?Điều\s*(\d+[a-z]?)', re.I)

    def clean(self, text):
        t = unicodedata.normalize('NFC', text)
        return t.strip()

    def split_amending_content(self, chunk, law_title):
        lines = chunk['content']['full_text'].split('\n')
        new_chunks = []
        
        current_target = "Lời dẫn/Mở đầu"
        current_lines = []
        
        print(f"   -> Đang xử lý nội dung Điều lớn: {chunk['metadata']['article_title']}")

        for line in lines:
            line_clean = self.clean(line)
            if not line_clean: continue

            # Kiểm tra xem dòng này có phải là bắt đầu một mục sửa đổi mới (VD: 1. Sửa đổi Điều 6)
            match = self.re_split_point.match(line_clean)
            if match:
                # Nếu đã có dữ liệu của mục trước đó, đóng gói lại
                if current_lines:
                    new_chunks.append(self.build_object(chunk, current_target, current_lines))
                
                # Bắt đầu mục mới
                current_target = f"Điều {match.group(2)}"
                current_lines = [line_clean]
                print(f"      [+] Tìm thấy mục sửa đổi cho: {current_target}")
            else:
                # Nạp dòng văn bản vào mục hiện tại
                current_lines.append(line_clean)

        # Đóng gói mục cuối cùng
        if current_lines:
            new_chunks.append(self.build_object(chunk, current_target, current_lines))
            
        return new_chunks

    def build_object(self, old_chunk, target_art, lines):
        # Tạo ID mới dựa trên ID cũ + số Điều bị tác động
        clean_target = re.sub(r'\D', '', target_art) if target_art else "0"
        return {
            "id": f"{old_chunk['id']}_Mod_D{clean_target}",
            "metadata": {
                **old_chunk['metadata'],
                "target_article": target_art,
                "is_sub_split": True
            },
            "content": {
                "full_text": "\n".join(lines)
            }
        }

    def run(self):
        if not os.path.exists(self.output_folder): os.makedirs(self.output_folder)
        files = [f for f in os.listdir(self.input_folder) if f.endswith('.json')]

        for f_name in files:
            path = os.path.join(self.input_folder, f_name)
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Heuristic: Dưới 15 chunks thường là luật sửa đổi
            if len(data) < 15:
                print(f"[*] Đang băm nhỏ file sửa đổi: {f_name}")
                refined_list = []
                for item in data:
                    # Nếu điều này chứa từ khóa sửa đổi thì mới băm
                    title = item['metadata']['article_title']
                    if "Sửa đổi" in title or "Bổ sung" in title:
                        refined_list.extend(self.split_amending_content(item, f_name))
                    else:
                        refined_list.append(item)
                
                final_output = refined_list
            else:
                print(f"[ ] Giữ nguyên file luật gốc: {f_name}")
                final_output = data

            # Lưu kết quả
            with open(os.path.join(self.output_folder, f_name), 'w', encoding='utf-8') as f:
                json.dump(final_output, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    SuperLawSplitter().run()