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
        # 1. FIX LINK: Xóa cái &Page=47 ở đây đi để biến này là link gốc sạch
        self.start_url = "https://vbpl.vn/bogiaoducdaotao/Pages/vanban.aspx?idLoaiVanBan=22&dvid=317"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        }
        self.session = requests.Session()

    def get_list_laws_by_page(self, page_number):
        laws = []
        connector = "&" if "?" in self.start_url else "?"
        url = f"{self.start_url}{connector}Page={page_number}"
            
        print(f"🔍 Đang 'đục' danh sách trang {page_number}...")
        
        try:
            # Lấy cookie trang chủ
            self.session.get(self.base_url, headers=self.headers, timeout=20)
            res = self.session.get(url, headers=self.headers, timeout=60)
            res.encoding = 'utf-8'
            
            if len(res.text) < 1000: 
                return []

            soup = BeautifulSoup(res.content, "html.parser")
            
            # Selector tìm các mục văn bản chuẩn cho trang Bộ GD
            items = soup.find_all(["div", "li", "tr"], class_=re.compile(r'doc-item|item|title', re.I))
            
            # Nếu không tìm thấy bằng class, thử tìm tất cả thẻ p có class title (giao diện hay dùng)
            if not items:
                items = soup.select('p.title')

            for li in items:
                status_text = li.get_text(" ", strip=True)
                # Lọc văn bản hết hiệu lực
                if "Hết hiệu lực toàn bộ" in status_text:
                    continue

                title_tag = li.find("a") if li.name != 'a' else li
                if title_tag and title_tag.get('href'):
                    title = title_tag.get_text(strip=True)
                    if len(title) < 20: continue 

                    # Logic lấy số hiệu
                    so_hieu = "Chua_Ro"
                    match = re.search(r'(?:Số|Số hiệu)[:\s]*([\d/]+[-A-ZĐ]+)', title, re.I)
                    if not match:
                        match = re.search(r'([\d/]+/[A-ZĐ-]+)', title)
                    
                    so_hieu = match.group(1).strip() if match else "Chua_Ro"
                    full_url = urljoin(self.base_url, title_tag['href'])
                    
                    if not any(l['url'] == full_url for l in laws):
                        laws.append({
                            "title": title,
                            "so_hieu": so_hieu,
                            "url": full_url
                        })
            
            print(f"✅ Trang {page_number}: Tìm thấy {len(laws)} văn bản còn hiệu lực.")
            return laws
            
        except Exception as e:
            print(f"❌ Lỗi trang {page_number}: {e}")
            return []

    # 2. 
    def run(self, start_page=60, max_page=65):
        output_dir = 'legal_data_json1'
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        total_crawled = 0
        current_page = start_page
        
        while current_page <= max_page:
            laws = self.get_list_laws_by_page(current_page)
            
            if not laws:
                print(f"🏁 Hết văn bản ở trang {current_page} hoặc bị lỗi. Dừng!")
                break
                
            for law in laws:
                clean_sh = re.sub(r'\W+', '_', law['so_hieu'])
                file_name = f"{clean_sh}.json"
                file_path = os.path.join(output_dir, file_name)
                
                if os.path.exists(file_path):
                    continue

                print(f"🚀 Đang xử lý: {law['so_hieu']}")
                try:
                    detail_url = law['url']
                    if "view=toanvan" not in detail_url:
                        detail_url += "&view=toanvan"
                        
                    res_detail = self.session.get(detail_url, headers=self.headers, timeout=30)
                    res_detail.encoding = 'utf-8'
                    
                    chunks = self.parse_chunks(res_detail.text, law)
                    if chunks:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(chunks, f, ensure_ascii=False, indent=2)
                        print(f"   [OK] Lưu {len(chunks)} điều.")
                        total_crawled += 1
                    
                    time.sleep(1) 
                except Exception as e:
                    print(f"   [Lỗi] {law['so_hieu']}: {e}")
            
            current_page += 1
            print(f"⏩ Chuyển sang trang {current_page}...")
            time.sleep(2)

        print(f"\n🎉 HOÀN THÀNH! Tổng cộng đã hốt được {total_crawled} văn bản.")

    def normalize_text(self, text):
        text = unicodedata.normalize('NFC', text)
        text = text.replace('\xa0', ' ').replace('\u200b', '')
        return re.sub(r'\s+', ' ', text).strip()

    def parse_chunks(self, html_content, meta):
        soup = BeautifulSoup(html_content, "html.parser")
        content_div = (soup.find("div", {"id": "toanvancontent"}) or 
                       soup.find("div", {"class": "toanvancontent"}) or
                       soup.find("div", {"id": "vbp-content"}))
        
        if not content_div: return []

        chunks = []
        cur_chap, cur_sec, cur_art_num, cur_art_title, cur_body = "Nội dung", "", "", "", []
        
        # Công tắc: Chỉ bắt đầu lấy khi gặp Mục I hoặc Điều 1
        started = False 

        # Regex bắt Điều (văn bản mới) hoặc Mục La Mã (văn bản cũ)
        re_art = re.compile(r'^[“"”\s]*Điều\s+(\d+[a-z]?)\s*[\.:-]', re.I)
        re_roman = re.compile(r'^([IVXLCDM]+)\.\s+(.*)', re.I)

        elements = content_div.find_all(['p', 'div', 'h3', 'h4'])
        
        for p in elements:
            if p.find(['p', 'div']): continue 
            text = self.normalize_text(p.get_text(" ", strip=True))
            if not text: continue

            is_bold = p.find(['strong', 'b'])
            art_match = re_art.match(text)
            roman_match = re_roman.match(text)

            # --- LOGIC BẮT ĐẦU TỪ MỤC I ---
            if not started:
                # Nếu gặp Mục I. hoặc Điều 1. thì mới bật công tắc
                if (roman_match and roman_match.group(1).upper() == "I") or (art_match and art_match.group(1) == "1"):
                    started = True
                else:
                    continue # Nếu chưa thấy thì cứ lướt qua mấy dòng rác ở đầu

            # Khi đã bắt đầu (started = True)
            if (art_match or roman_match) and is_bold:
                # Lưu chunk cũ trước khi sang cái mới
                if cur_art_num:
                    chunks.append(self.save_chunk(meta, cur_chap, cur_sec, cur_art_num, cur_art_title, cur_body))
                
                # Cập nhật thông tin mới
                if art_match:
                    cur_art_num = art_match.group(1)
                else:
                    cur_art_num = roman_match.group(1) # Lấy I, II, III làm Article Number
                
                cur_art_title = text
                cur_body = [text]
                cur_chap = text # Cho tiêu đề mục vào Chapter luôn cho dễ nhìn
            
            else:
                # Nếu đang trong một mục thì gom text vào body
                if cur_art_num:
                    cur_body.append(text)

        # Lưu thằng cuối cùng
        if cur_art_num:
            chunks.append(self.save_chunk(meta, cur_chap, cur_sec, cur_art_num, cur_art_title, cur_body))
        
        return chunks
    
    

    def save_chunk(self, meta, chap, sec, num, title, body):
        prefix = re.sub(r'\W+', '_', meta['so_hieu'])
        return {
            "id": f"{prefix}_D{num}",
            "metadata": {
                "source": meta['title'],
                "so_hieu": meta['so_hieu'],
                "chapter": chap,
                "section": sec,
                "article_number": num,
                "article_title": title,
            },
            "content": {
                "full_text": "\n".join(body)
            }
        }

if __name__ == "__main__":
    crawler = LawCrawlerUltimate()
    # 3. CHỈ CẦN GỌI THẾ NÀY:
    crawler.run(start_page=60, max_page=65)