import json
import os
import sys

# Define the ontology
ontology = {
    "intent_classifier": {
        "keywords": {
            "GREETING": ["xin chào", "chào bạn", "hello", "hi ", "hey", "chào"],
            "THANKS": ["cảm ơn", "thank", "cám ơn", "tks"],
            "SUMMARY": ["tóm tắt", "khái quát", "nội dung chính", "quy định những gì"],
            "STATISTICAL": ["bao nhiêu", "số lượng", "đếm", "thống kê"],
            "LISTING": ["liệt kê", "tất cả", "những trường hợp nào", "trường hợp nào", "những đối tượng nào", "đối tượng nào", "bao gồm những gì", "các loại", "kể tên"],
            "COMPARISON": ["so sánh", " vs ", "đối chiếu"],
            "UNANSWERABLE": ["top 100", "thế giới", "xếp hạng quốc tế"]
        },
        "lookup_context": [
            "bồi hoàn", "mức", "phí", "lương", "trợ cấp", "hỗ trợ",
            "phạt", "tiền", "chi phí", "học phí", "tuổi", "học sinh",
            "lớp", "lâu", "năm", "tháng", "ngày", "tối đa", "tối thiểu",
            "giờ", "tiết", "điểm", "hạng"
        ],
        "legal_keywords": [
            "điều", "khoản", "điểm", "luật", "nghị định", "thông tư", "quy định",
            "quyết định", "văn bản", "pháp luật", "hiến pháp", "quyền", "nghĩa vụ",
            "điều kiện", "tiêu chuẩn", "trách nhiệm", "xử phạt", "vi phạm",
            "hướng dẫn", "thi hành", "hiệu lực", "bãi bỏ", "sửa đổi", "bổ sung",
            "ban hành", "áp dụng", "chuyển đổi", "quyết nghị",
            "giáo viên", "nhà giáo", "giảng viên", "học sinh", "sinh viên",
            "trường", "đại học", "mầm non", "tiểu học", "trung học", "thcs", "thpt",
            "giáo dục", "đào tạo", "tuyển sinh", "học phí", "học bổng",
            "sư phạm", "chức danh", "nghề nghiệp", "thăng hạng", "bổ nhiệm",
            "hội đồng", "hiệu trưởng", "cơ sở giáo dục", "chương trình",
            "bằng cấp", "tốt nghiệp", "kỷ luật", "đánh giá", "kiểm tra",
            "nâng chuẩn", "du học", "cử tuyển", "bồi hoàn", "thỉnh giảng",
            "lợi nhuận", "tư thục", "công lập", "hỗ trợ", "chính sách",
            "khu công nghiệp", "dân tộc", "nội trú"
        ]
    },
    "target_routing_rules": [
        { "keywords": ["nâng chuẩn", "trình độ chuẩn", "bằng cử nhân giáo viên"], "targets": ["71/2020/NĐ-CP"] },
        { "keywords": ["bổ nhiệm hạng", "thăng hạng", "hạng ii", "hạng iii"], "conditions": [
            { "requires": ["mầm non"], "targets": ["01/2021/TT-BGDĐT", "08/2023/TT-BGDĐT"] },
            { "requires": ["tiểu học"], "targets": ["02/2021/TT-BGDĐT", "08/2023/TT-BGDĐT"] },
            { "requires": [], "targets": ["01/2021/TT-BGDĐT", "08/2023/TT-BGDĐT"] }
        ]},
        { "keywords": ["giữa kỳ", "cuối kỳ", "bài kiểm tra", "kiểm tra định kỳ", "kiểm tra thường xuyên", "nghỉ học nhiều", "ở lại lớp", "lên lớp", "tai nạn", "kiểm tra giữa kỳ", "nghỉ học quá 45 buổi"], "targets": ["22/2021/TT-BGDĐT"] },
        { "keywords": ["chuyển đổi"], "conditions": [{ "requires": ["đại học"], "targets": ["34/2018/QH14", "99/2019/NĐ-CP"] }] },
        { "keywords": ["đại học tư thục", "không vì lợi nhuận", "hội đồng trường", "đại học y"], "targets": ["34/2018/QH14", "99/2019/NĐ-CP"] },
        { "keywords": ["thi tốt nghiệp", "tốt nghiệp thpt", "2026", "24/2024"], "targets": ["24/2024/TT-BGDĐT"] },
        { "keywords": ["tuyển sinh năm 2022", "tuyển sinh 2022", "tuyển sinh đại học"], "targets": ["08/2022/TT-BGDĐT"] },
        { "keywords": ["học bổng", "cử tuyển", "dự bị đại học", "nội trú"], "targets": ["84/2020/NĐ-CP"] },
        { "keywords": ["du học", "nghiên cứu khoa học nước ngoài", "gia hạn du học", "sinh viên cử đi học", "giảng viên đại học cử ra nước ngoài"], "targets": ["86/2021/NĐ-CP"] },
        { "keywords": ["sinh viên sư phạm", "hỗ trợ sinh hoạt phí", "bồi hoàn", "116/2020", "bảo lưu sư phạm"], "targets": ["116/2020/NĐ-CP"] },
        { "keywords": ["chuyển trường khác tỉnh", "giấy giới thiệu", "thủ tục chuyển trường"], "targets": ["43/2019/QH14"] },
        { "keywords": ["ngoại ngữ", "dạy ngoại ngữ", "ngoại ngữ 1", "chương trình giáo dục phổ thông 2018", "chương trình 2018"], "targets": ["32/2018/TT-BGDĐT"] },
        { "keywords": ["ngày pháp luật", "ngày pháp luật việt nam"], "targets": ["14/2012/QH13"] },
        { "keywords": ["học phí", "tiểu học", "không phải đóng học phí", "trường công lập"], "targets": ["43/2019/QH14"], "match_type": "any" },
        { "keywords": ["giảng viên", "chuẩn giảng viên", "chuẩn tối thiểu"], "targets": ["34/2018/QH14"], "match_type": "any" },
        { "keywords": ["chế độ làm việc", "nghỉ hè hằng năm", "nghỉ hè của nhà giáo", "tuổi nghỉ hưu của nhà giáo"], "targets": ["Luật 73/2025/QH15"], "match_type": "any" },
        { "keywords": ["hòa nhập", "giáo dục hòa nhập", "trung tâm hỗ trợ", "phạm vi điều chỉnh"], "targets": ["20/2022/TT-BGDĐT"] },
        { "keywords": ["nhiệm kỳ", "hiệu trưởng", "nhiệm kỳ của hiệu trưởng"], "targets": ["08/2012/QH13", "34/2018/QH14"] },
        { "keywords": ["quốc phòng", "an ninh", "quốc phòng và an ninh", "chính khóa"], "targets": ["30/2013/QH13"] },
        { "keywords": ["giảm 02 tiết", "giảm 2 tiết", "hội đồng trường", "kiêm nhiệm chủ tịch"], "targets": ["05/2025/TT-BGDĐT"] },
        { "keywords": ["05 năm ở nước ngoài", "ít nhất 05 năm", "liên kết giáo dục", "bên nước ngoài"], "targets": ["124/2024/NĐ-CP", "202/2025/NĐ-CP"] },
        { "keywords": ["sáp nhập", "chia, tách", "chia tách trường tiểu học", "ubnd cấp huyện"], "targets": ["07/BGDĐT-VBHN"] },
        { "keywords": ["trường học an toàn", "tai nạn thương tích"], "conditions": [
            { "requires": ["mầm non"], "targets": ["45/2021/TT-BGDĐT"] },
            { "requires": ["phổ thông", "thường xuyên", "thcs", "thpt", "tiểu học", "cấp 1", "cấp 2", "cấp 3"], "match_type": "any", "targets": ["18/2023/TT-BGDĐT"] }
        ]}
    ]
}

# Append expansion rules dynamically from the actual code
sys.path.insert(0, r"d:\legal-edu-chatbot")
from app.query.query_expander import EducationQueryExpander
expansion_rules = []
for kws, exp in EducationQueryExpander.EXPANSION_RULES:
    expansion_rules.append({"keywords": kws, "expansion": exp})
ontology["expansion_rules"] = expansion_rules

with open(r"d:\legal-edu-chatbot\data\domain_knowledge.json", "w", encoding="utf-8") as f:
    json.dump(ontology, f, ensure_ascii=False, indent=4)

print("Ontology generated successfully!")
