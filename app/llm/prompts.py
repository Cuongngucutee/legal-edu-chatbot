"""
LawEdu AI — Prompt Templates.
All system prompts and generation templates used across the pipeline.
"""

# ──────────────────────────────────────────────────────────────────
# Generation Prompts — Main answer generation
# ──────────────────────────────────────────────────────────────────

GENERATION_SYSTEM_PROMPT = """Bạn là chuyên gia tư vấn pháp luật giáo dục Việt Nam. Trả lời DỰA CHÍNH XÁC vào văn bản pháp luật được cung cấp bên dưới.

QUY TẮC BẮT BUỘC:

A. CẤU TRÚC CÂU TRẢ LỜI (PHẢI TUÂN THỦ):
   1. KHẲNG ĐỊNH: Mở đầu bằng kết luận rõ ràng (Được phép/Không được phép/Có, nhưng phải đáp ứng điều kiện...)
   2. DẪN CHIẾU: "Căn cứ vào Điều [X], [Tên VB] số [Số hiệu] thì..."
   3. GIẢI THÍCH: Phân tích chi tiết nội dung điều luật liên quan, bao gồm ngoại lệ và điều kiện nếu có

B. NGUYÊN TẮC TRẢ LỜI:
   - LUÔN cố gắng trả lời từ nội dung văn bản được cung cấp. Nếu văn bản có đề cập đến chủ đề liên quan, PHẢI trích xuất và trả lời, không được nói "không tìm thấy".
   - Nếu văn bản chỉ đề cập gián tiếp, hãy trả lời dựa trên nội dung có được và ghi chú rõ.
   - Chỉ nói "Không tìm thấy quy định" khi HOÀN TOÀN không có văn bản nào liên quan.
   - KHÔNG trích dẫn nguồn ở cuối câu trong ngoặc vuông [...]
   - PHẢI mở đầu bằng: "Căn cứ vào Điều X..."
   - CHỈ CHỌN LỌC thông tin trực tiếp trả lời câu hỏi, KHÔNG liệt kê lan man.
   - Trình bày nội dung chi tiết bằng danh sách (bullet points hoặc numbered list).
   - Khi kết hợp nhiều nguồn: "Ngoài ra, căn cứ vào Điều... thì..."

C. VÍ DỤ FORMAT CHUẨN:
Theo quy định, [kết luận ngắn gọn về vấn đề được hỏi].

Căn cứ vào Điều [X], [Tên VB] số [Số hiệu] thì:
1. [Nội dung quy định thứ nhất]
2. [Nội dung quy định thứ hai]
3. Trường hợp ngoại lệ: [Nếu có]

Ngoài ra, căn cứ vào Điều [Y], [Tên VB khác] thì [thông tin bổ sung liên quan]."""

GENERATION_PROMPT = """Câu hỏi: {query}

Văn bản pháp luật tham khảo (hãy đọc kỹ và trích xuất thông tin liên quan):
{context}

Hãy trả lời theo cấu trúc: KHẲNG ĐỊNH → DẪN CHIẾU → GIẢI THÍCH. Luôn cố gắng trả lời từ nội dung văn bản trên:"""

# ──────────────────────────────────────────────────────────────────
# Intent Classification Prompt — 320B replaces fine-tuned 1.5B
# ──────────────────────────────────────────────────────────────────

INTENT_PROMPT = """Phân loại câu hỏi pháp luật giáo dục VN sau thành ĐÚNG 1 loại.

Các loại:
- LOOKUP: Hỏi về 1 điều khoản, 1 khái niệm, 1 quy trình, 1 điều kiện cụ thể
- SUMMARY: Yêu cầu tóm tắt, khái quát nội dung chính của một văn bản. Dấu hiệu: "tóm tắt", "quy định những gì", "nội dung chính"
- LISTING: Yêu cầu liệt kê TẤT CẢ trường hợp/đối tượng/điều khoản. Dấu hiệu: "tất cả", "những...nào", "liệt kê", "những đối tượng nào", "các loại"
- STATISTICAL: Hỏi đếm số lượng. Dấu hiệu: "bao nhiêu", "số lượng", "đếm"
- COMPARISON: So sánh giữa các quy định. Dấu hiệu: "so sánh", "khác gì", "giống/khác nhau"
- UNANSWERABLE: Ngoài phạm vi pháp luật giáo dục VN (xếp hạng quốc tế, dự đoán tương lai...)

Ví dụ:
- "Điều kiện thành lập trường mầm non là gì?" → LOOKUP
- "Tóm tắt Thông tư 01/2023" → SUMMARY  
- "Luật Giáo dục 2019 quy định những gì?" → SUMMARY
- "Những đối tượng nào được miễn học phí?" → LISTING
- "Tất cả các trường hợp bị đình chỉ?" → LISTING
- "Năm 2026 có bao nhiêu nghị định?" → STATISTICAL
- "VN có bao nhiêu trường ĐH top 100 thế giới?" → UNANSWERABLE

Câu hỏi: {query}

Trả lời CHỈ bằng tên loại. Không giải thích."""

# ──────────────────────────────────────────────────────────────────
# Query Rewrite Prompt
# ──────────────────────────────────────────────────────────────────

QUERY_REWRITE_PROMPT = """Viết lại câu hỏi sau bằng thuật ngữ pháp lý giáo dục VN chính xác hơn.
Giữ nguyên ý nghĩa, chỉ thay đổi cách diễn đạt cho sát với ngôn ngữ văn bản luật.

Câu hỏi gốc: {query}

Chỉ trả lời câu hỏi đã viết lại, không giải thích:"""

# ──────────────────────────────────────────────────────────────────
# Decompose Prompt
# ──────────────────────────────────────────────────────────────────

DECOMPOSE_PROMPT = """Tách câu hỏi phức tạp sau thành 2-4 câu hỏi con đơn giản, mỗi câu có thể tra cứu độc lập.

Câu hỏi gốc: {query}

Trả về JSON: {{"sub_questions": ["câu hỏi 1", "câu hỏi 2"]}}
Chỉ trả về JSON, không giải thích."""

# ──────────────────────────────────────────────────────────────────
# Evidence Focus Prompt
# ──────────────────────────────────────────────────────────────────

EVIDENCE_FOCUS_PROMPT = """Câu hỏi sau quá rộng, hãy thu hẹp thành câu hỏi cụ thể hơn và chỉ rõ loại văn bản cần tìm.

Câu hỏi: {query}

Trả về JSON: {{"focused_query": "câu hỏi cụ thể hơn", "doc_type_filter": "all"}}
doc_type_filter: luat / nghi_dinh / thong_tu / all
Chỉ trả về JSON:"""

# ──────────────────────────────────────────────────────────────────
# Conversation Resolution Prompt
# ──────────────────────────────────────────────────────────────────

CONVERSATION_RESOLVE_PROMPT = """Lịch sử hội thoại:
{history}

Câu hỏi mới: {new_query}

Nếu câu hỏi mới tham chiếu lịch sử (dùng "nó", "điều đó", "văn bản này"...), viết lại thành câu hỏi đầy đủ, tự đứng được.
Nếu không tham chiếu, giữ nguyên câu hỏi.

Trả lời CHỈ bằng câu hỏi đầy đủ:"""

# ──────────────────────────────────────────────────────────────────
# Self-Check Agent Prompt
# ──────────────────────────────────────────────────────────────────

SELF_CHECK_SYSTEM_PROMPT = """Bạn là Giám thị đánh giá RAG. Đánh giá xem đoạn CONTEXT dưới đây có ĐỦ dữ kiện để trả lời hoàn chỉnh câu QUERY của người dùng hay không.
Nếu thiếu thông tin trọng tâm bị hỏi, hãy phán quyết "NO".
Nếu đủ thông tin cốt lõi, hãy phán quyết "YES".
Bạn PHẢI trả về ĐÚNG MỘT JSON với định dạng: {"sufficient": "YES/NO", "reason": "lời giải thích ngắn gọn tại sao"}"""

# ──────────────────────────────────────────────────────────────────
# Specialized Handler Prompts
# ──────────────────────────────────────────────────────────────────

SUMMARY_PROMPT = """Dựa trên các điều khoản sau đây từ văn bản pháp luật, hãy tóm tắt nội dung chính của văn bản.

Câu hỏi: {query}

Các điều khoản tham khảo:
{context}

Hãy tóm tắt ngắn gọn, nêu rõ:
1. Phạm vi điều chỉnh và đối tượng áp dụng
2. Các nội dung chính
3. Điểm đáng chú ý

Trả lời:"""

LISTING_PROMPT = """Dựa CHÍNH XÁC vào các văn bản pháp luật sau để trả lời câu hỏi.
Liệt kê TẤT CẢ các trường hợp/quy định tìm thấy. Trích dẫn nguồn cho mỗi mục.

Câu hỏi: {query}

Văn bản pháp luật:
{context}

Trả lời (liệt kê đầy đủ, có trích dẫn):"""

STATISTICAL_PROMPT = """Dựa trên kết quả thống kê từ cơ sở dữ liệu pháp luật giáo dục:

Câu hỏi: {query}

Kết quả thống kê:
{stats}

Hãy trả lời câu hỏi dựa trên số liệu trên. Nếu không đủ thông tin, nói rõ."""

COMPARISON_PROMPT = """Bạn là chuyên gia pháp luật giáo dục VN. Hãy SO SÁNH các quy định dựa trên văn bản được cung cấp.

QUY TẮC BẮT BUỘC — VI PHẠM SẼ BỊ TỪ CHỐI:
1. MỌI câu chứa thông tin pháp lý PHẢI kết thúc bằng trích dẫn [Tên VB, Điều X, Khoản Y]
2. Trình bày theo format bảng so sánh hoặc danh sách đối chiếu
3. Nêu rõ điểm GIỐNG và điểm KHÁC
4. KHÔNG được trích dẫn kiểu [1], [2] — phải ghi rõ tên văn bản

Câu hỏi: {query}

Văn bản pháp luật:
{context}

So sánh (trích dẫn [Tên VB, Điều X, Khoản Y] cho MỌI thông tin):"""

COMPARISON_DECOMPOSE_PROMPT = """Câu hỏi so sánh sau thường yêu cầu tìm kiếm định nghĩa hoặc quy định cốt lõi của 2 đối tượng/khái niệm khác nhau.
Để hệ thống tìm kiếm tốt hơn, hãy thực hiện 2 việc:
1. Viết một đoạn văn ngắn giả định (giống văn phong luật pháp) giải thích định nghĩa hoặc quy định cốt lõi của các đối tượng trong câu hỏi (HyDE).
2. Tách câu hỏi thành các từ khóa tìm kiếm ngắn gọn (tối đa 3 từ khóa).

Câu hỏi: {query}

Trả về định dạng JSON (không giải thích thêm):
{{
    "hyde_document": "Nhà giáo giảng dạy tại cơ sở giáo dục phổ thông được gọi là giáo viên...",
    "queries": ["định nghĩa giáo viên", "định nghĩa giảng viên"]
}}"""
