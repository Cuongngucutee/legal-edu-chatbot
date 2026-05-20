import requests
from bs4 import BeautifulSoup
import json
import re
import os
import time
import unicodedata
from urllib.parse import urljoin

class LawCrawlerUltimate:
    def __init__(self):
        self.base_url = "https://vbpl.vn"
        self.start_url = "https://vbpl.vn/bogiaoducdaotao/Pages/vanban.aspx?idLoaiVanBan=17&dvid=317"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }
        self.session = requests.Session()

    def get_list_laws(self):
        laws = []
        url = self.start_url
        page = 1
        while url:
            print(f"--- Đang quét danh sách trang {page}... ---")
            res = self.session.get(url, headers=self.headers)
            soup = BeautifulSoup(res.content, "html.parser")
            items = soup.select("ul.listLaw > li")
            for li in items:
                status = li.select_one(".right p.red")
                if status and "Hết hiệu lực toàn bộ" in status.get_text():
                    continue
                title_tag = li.select_one("p.title a")
                if title_tag:
                    laws.append({
                        "title": title_tag.get_text(strip=True),
                        "url": urljoin(self.base_url, title_tag['href'])
                    })
            next_btn = soup.find("a", string=str(page + 1))
            if next_btn:
                url = urljoin(self.base_url, next_btn['href'])
                page += 1
            else:
                url = None
        return laws

    def normalize_text(self, text):
        """Chuẩn hóa Unicode và dọn dẹp khoảng trắng"""
        text = unicodedata.normalize('NFC', text)
        text = text.replace('\xa0', ' ').replace('\u200b', '')
        return re.sub(r'\s+', ' ', text).strip()

    def parse_chunks(self, html_content, meta):
        soup = BeautifulSoup(html_content, "html.parser")
        content_div = soup.find("div", {"id": "toanvancontent"}) or soup.find("div", {"class": "toanvancontent"})
        if not content_div:
            return []

        # Xử lý: Hợp nhất nội dung trong mỗi thẻ p/div để không bị chia cắt "Điều 1"
        processed_lines = []
        for p in content_div.find_all(['p', 'div', 'h1', 'h2', 'h3', 'h4']):
            # get_text() với strip=True sẽ tự nối các thẻ strong/a/span bên trong p
            text = p.get_text(" ", strip=True) 
            clean_text = self.normalize_text(text)
            if clean_text:
                processed_lines.append(clean_text)

        chunks = []
        cur_chap = "Chương I"
        cur_sec = ""
        cur_art_num = ""
        cur_art_title = ""
        cur_body = []

        # Regex nhận diện (Linh hoạt cho cả "Điều 1." và "Điều 1:")
        re_chap = re.compile(r'^Chương\s+([IVXLCDM\d]+)', re.I)
        re_sec = re.compile(r'^Mục\s+(\d+)', re.I)
        re_art = re.compile(r'^Điều\s+(\d+[a-z]?)\s*[\.:-]', re.I)

        for line in processed_lines:
            # Kiểm tra Chương/Mục để cập nhật metadata
            if re_chap.match(line):
                cur_chap = line
                continue
            if re_sec.match(line):
                cur_sec = line
                continue
            
            # Kiểm tra bắt đầu một Điều mới
            art_match = re_art.match(line)
            if art_match:
                # Nếu đã có Điều trước đó trong bộ nhớ, lưu nó lại
                if cur_art_num:
                    chunks.append(self.save_chunk(meta, cur_chap, cur_sec, cur_art_num, cur_art_title, cur_body))
                
                # Khởi tạo Điều mới
                cur_art_num = art_match.group(1)
                cur_art_title = line
                cur_body = [line]
            else:
                # Nếu đang trong một Điều, nạp nội dung vào body
                if cur_art_num:
                    cur_body.append(line)

        # Lưu Điều cuối cùng của văn bản
        if cur_art_num:
            chunks.append(self.save_chunk(meta, cur_chap, cur_sec, cur_art_num, cur_art_title, cur_body))
        
        return chunks

    def save_chunk(self, meta, chap, sec, num, title, body):
        # Tạo ID từ số hiệu luật
        prefix = re.sub(r'\W+', '', meta['title'].split('/')[0])
        return {
            "id": f"{prefix}_D{num}",
            "metadata": {
                "source": meta['title'],
                "chapter": chap,
                "section": sec,
                "article_number": num,
                "article_title": title,
                "tags": ["pháp luật giáo dục", "văn bản quy phạm"]
            },
            "content": {
                "full_text": "\n".join(body)
            }
        }

    def run(self):
        output_dir = 'legal_data_json'
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        laws = self.get_list_laws()
        print(f"Hệ thống tìm thấy {len(laws)} văn bản còn hiệu lực. Bắt đầu xử lý...")

        for law in laws:
            print(f"-> Đang xử lý: {law['title']}")
            try:
                res = self.session.get(law['url'], headers=self.headers, timeout=20)
                res.encoding = 'utf-8'
                
                chunks = self.parse_chunks(res.text, law)
                
                if chunks:
                    # Tạo tên file an toàn (loại bỏ dấu gạch chéo)
                    safe_name = re.sub(r'[/\\:*?"<>|]', '_', law['title'])
                    with open(f"{output_dir}/{safe_name}.json", 'w', encoding='utf-8') as f:
                        json.dump(chunks, f, ensure_ascii=False, indent=2)
                    print(f"   [Thành công] Đã tách được {len(chunks)} điều.")
                else:
                    print(f"   [Cảnh báo] Không tìm thấy dữ liệu điều nào.")
                
                time.sleep(1)
            except Exception as e:
                print(f"   [Lỗi] {e}")

if __name__ == "__main__":
    LawCrawlerUltimate().run()