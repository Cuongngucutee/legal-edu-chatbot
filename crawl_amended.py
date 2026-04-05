import json
import os
import re
import unicodedata

class SuperLawSplitter:
    def __init__(self, input_folder='legal_data_json1', output_folder='final_refined_data1'):
        self.input_folder = input_folder
        self.output_folder = output_folder
        # REGEX MỚI: Bắt cả Điều, Chương, Mục bị sửa đổi/bổ sung
        # Nó sẽ bắt được: "1. Sửa đổi Điều 5", "2. Bổ sung Chương III", "3. Bãi bỏ Mục 1"
        self.re_split_point = re.compile(
            r'^(\d+)[\.\)]\s*(?:Sửa đổi|Bổ sung|Bãi bỏ|Thay thế).*?(Điều|Chương|Mục)\s*([IVXLCDM\d]+[a-z]?)', 
            re.I
        )

    def clean(self, text):
        t = unicodedata.normalize('NFC', text)
        return t.strip()

    def split_amending_content(self, chunk, law_title):
        lines = chunk['content']['full_text'].split('\n')
        new_chunks = []
        
        current_target = "Lời dẫn/Mở đầu"
        current_lines = []
        
        print(f"   -> Đang xẻ nhỏ nội dung sửa đổi: {chunk['metadata']['article_title']}")

        for line in lines:
            line_clean = self.clean(line)
            if not line_clean: continue

            # Kiểm tra xem dòng này có phải là bắt đầu một phần sửa đổi mới không
            match = self.re_split_point.match(line_clean)
            
            if match:
                # Nếu đã có dữ liệu của phần trước đó, đóng gói lại
                if current_lines:
                    new_chunks.append(self.build_object(chunk, current_target, current_lines))
                
                # Bắt đầu phần mới: VD "Điều 5", "Chương II", "Mục 1"
                type_node = match.group(2).capitalize() # Điều/Chương/Mục
                value_node = match.group(3)             # 5 / II / 1
                current_target = f"{type_node} {value_node}"
                
                current_lines = [line_clean]
                print(f"      [+] Phát hiện sửa đổi cho: {current_target}")
            else:
                # Nạp dòng văn bản vào phần hiện tại
                current_lines.append(line_clean)

        # Đóng gói phần cuối cùng
        if current_lines:
            new_chunks.append(self.build_object(chunk, current_target, current_lines))
            
        return new_chunks

    def build_object(self, old_chunk, target_node, lines):
        # Tạo ID mới dựa trên ID cũ + (Chương/Mục/Điều) bị tác động
        # VD: ..._Mod_Chuong_II hoặc ..._Mod_Dieu_5
        safe_target = re.sub(r'\s+', '_', target_node) # Thay dấu cách bằng gạch nối
        safe_target = re.sub(r'[^a-zA-Z0-9_]', '', safe_target) # Dọn ký tự lạ
        
        return {
            "id": f"{old_chunk['id']}_Mod_{safe_target}",
            "metadata": {
                **old_chunk['metadata'],
                "target_node": target_node, # Đổi tên cho tổng quát (không chỉ là article)
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