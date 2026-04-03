import re

class RuleEngine:
    def __init__(self, book_index=None):
        self.book_index = book_index
        # Bộ từ điển để xác định tính hợp lệ của chủ đề (Domain Education Law)
        self.domain_keywords = [
            "luật", "nghị định", "thông tư", "quy định", "điều", "khoản", "giáo dục", 
            "đại học", "phổ thông", "trường", "sinh viên", "học sinh", "giảng viên", "giáo viên",
            "tuyển sinh", "đào tạo", "chương trình", "bộ", "phạt", "kỷ luật", "khen thưởng",
            "học phí", "đình chỉ", "cơ sở", "nhà giáo", "bằng", "chứng chỉ", "thủ tục",
            "biên chế", "nhà nước", "chính phủ", "tiêu chuẩn", "đánh giá", "lớp", "môn",
            "xét", "học", "dạy", "thi", "kiểm tra", "điểm"
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
        has_domain = any(k in query_lower for k in self.domain_keywords)
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
