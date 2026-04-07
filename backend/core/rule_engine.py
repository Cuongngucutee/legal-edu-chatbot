import re

class RuleEngine:
    def __init__(self, book_index=None):
        self.book_index = book_index
        # Bộ từ điển được phân loại theo từng mảng trong Luật Giáo dục
        self.domain_keywords = {
            "van_ban_phap_luat": [
                "luật giáo dục", "nghị định", "thông tư", "quyết định", "quy chế", 
                "quy định", "điều", "khoản", "điểm", "ban hành", "sửa đổi", "bổ sung", 
                "hiệu lực", "thi hành", "bãi bỏ", "hướng dẫn", "công văn", "chỉ thị"
            ],
            "cap_bac_va_co_so_dao_tao": [
                "mầm non", "mẫu giáo", "tiểu học", "trung học cơ sở", "thcs", 
                "trung học phổ thông", "thpt", "đại học", "cao đẳng", "trung cấp", 
                "nghề nghiệp", "sau đại học", "thạc sĩ", "tiến sĩ", "giáo dục thường xuyên", 
                "gdtx", "công lập", "dân lập", "tư thục", "cơ sở giáo dục"
            ],
            "chu_the_nhan_su": [
                "bộ giáo dục", "bgdđt", "sở giáo dục", "phòng giáo dục", "nhà nước", 
                "chính phủ", "hiệu trưởng", "giáo viên", "giảng viên", "nhà giáo", 
                "người học", "học sinh", "sinh viên", "học viên", "cán bộ quản lý", 
                "thanh tra giáo dục", "hội đồng trường"
            ],
            "hoat_dong_chuyen_mon": [
                "tuyển sinh", "đào tạo", "chương trình giáo dục", "sách giáo khoa", 
                "giáo trình", "chuẩn đầu ra", "kiểm định chất lượng", "đánh giá", 
                "thi tốt nghiệp", "xét tuyển", "tín chỉ", "niên chế", "chuyển trường", 
                "lưu ban", "học bạ", "văn bằng", "chứng chỉ"
            ],
            "hanh_chinh_va_che_tai": [
                "học phí", "miễn giảm", "học bổng", "trợ cấp", "biên chế", "hợp đồng làm việc",
                "kỷ luật", "khen thưởng", "xử phạt hành chính", "đình chỉ học tập", 
                "buộc thôi học", "cảnh cáo", "khiển trách", "khiếu nại", "tố cáo"
            ]
        }

        # Tạo một list phẳng (flattened list) để dùng cho các hàm kiểm tra đơn giản
        self.flat_keywords = [
            word for category in self.domain_keywords.values() for word in category
        ]
        
        # Danh sách các từ đơn phổ thông có thể gây nhiễu, cần cẩn thận khi đối chiếu
        self.risky_single_words = [
            "xét", "học", "dạy", "thi", "điểm", "lớp", "môn", "bộ", "cơ sở"
        ]

    def validate(self, query: str) -> dict:
        """
        Thực thi các luật đánh giá cứng tốn 0ms, tiết kiệm API (Pre-gate).
        Trả về json {pass: bool, reason: str}
        """
        # 1. Format check (Độ dài)
        if len(query) < 8:
            return {"pass": False, "reason": "Câu hỏi của bạn quá ngắn. Bạn vui lòng cung cấp câu hỏi đầy đủ chủ ngữ hoặc chi tiết hơn nhé!"}
        if len(query) > 1000:
            return {"pass": False, "reason": "Câu hỏi quá dài (Vượt 1000 ký tự). Bạn vui lòng tóm tắt thông điệp gọn gàng hơn giúp tôi."}

        # 2. Out-of-scope Domain verification
        query_lower = query.lower()
        has_domain = any(k in query_lower for k in self.flat_keywords)
        
        if not has_domain:
            # Check nhóm từ dễ gây nhiễu bằng cách bắt buộc nó phải là nguyên chữ (Word Boundary)
            words = set(re.findall(r'\w+', query_lower))
            has_domain = any(w in words for w in self.risky_single_words)
            
        if not has_domain:
            return {"pass": False, "reason": "Rất tiếc, có vẻ câu hỏi của bạn không bám sát nội dung **Pháp luật & Giáo dục**. Xin hãy đặt câu hỏi đúng phạm vi chuyên môn của tôi."}

        # 3. Cite validator (Check tính hão huyền của điều luật - nếu book index được truyền)
        match = re.search(r'điều\s+(\d+)', query_lower)
        if match and self.book_index:
            article_num = match.group(1)
            # Quét nhanh mã d[article_num] xem có tồn tại trong cây không
            found = False
            for node_id in self.book_index.graph.nodes:
                if str(node_id).endswith(f":d{article_num}"):
                    found = True
                    break
            if not found:
                return {
                    "pass": False, 
                    "reason": f"Dựa trên CSDL tôi đang nắm giữ, không tìm thấy văn bản nào quy định cụ thể về **Điều {article_num}** mà bạn vừa nhắc tới. Bạn vui lòng kiểm tra lại sự tồn tại của Điều luật này nhé."
                }
                
        return {"pass": True, "reason": "OK"}
