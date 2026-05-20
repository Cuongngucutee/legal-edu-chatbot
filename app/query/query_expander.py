"""
LawEdu AI — Education Query Expander.
Expands common education queries into legal terminology for better retrieval.
No LLM needed — pure keyword mapping rules.
"""


class EducationQueryExpander:
    """Expand câu hỏi giáo dục phổ thông bằng keyword mapping trước khi retrieve.

    Vấn đề: Người dùng hỏi "điện thoại di động" nhưng chunk pháp lý nằm trong
    "Điều lệ trường THCS THPT" → embedding miss. Expander bổ sung thuật ngữ pháp lý
    để tăng recall mà không cần LLM call.
    """

    EXPANSION_RULES = [
        # Tuổi vào lớp 1 → Luật 43/2019
        (["tuổi vào lớp 1", "tuổi vào học", "nhập học lớp 1"],
         "tuổi của học sinh vào học lớp 1 là 6 tuổi phát triển sớm trí tuệ Luật Giáo dục 43/2019 Điều 28"),

        # Hồ sơ sổ sách giáo viên
        (["sổ sách", "hồ sơ giáo viên tiểu học", "hồ sơ quản lý", "giáo án"],
         "hồ sơ quản lý hoạt động giáo dục kế hoạch bài dạy sổ ghi chép"),

        # Tiểu học chung
        (["tiểu học", "lớp 1", "lớp 2", "lớp 3", "lớp 4", "lớp 5"],
         "quy định trường tiểu học giáo dục tiểu học"),

        # THCS/THPT
        (["thcs", "thpt", "trung học cơ sở", "trung học phổ thông", "cấp 2", "cấp 3"],
         "trường trung học cơ sở trung học phổ thông quy định"),

        # Điện thoại di động
        (["điện thoại", "điện thoại di động", "sử dụng điện thoại"],
         "học sinh sử dụng điện thoại di động trong giờ học an sinh số"),

        # Sĩ số lớp học
        (["sĩ số", "tối đa bao nhiêu học sinh", "số học sinh trong lớp"],
         "số lượng học sinh trong một lớp sĩ số tối đa quy định"),

        # Kỷ luật học sinh
        (["kỷ luật", "đình chỉ", "đuổi học", "kỷ luật học sinh", "kỷ luật đình chỉ"],
         "hình thức kỷ luật học sinh đình chỉ học tập thời hạn"),

        # Chuyển trường
        (["chuyển trường", "chuyển trường khác tỉnh"],
         "chuyển trường tiếp nhận học sinh giấy giới thiệu thủ tục chuyển trường"),

        # Tốt nghiệp THCS
        (["tốt nghiệp thcs", "xét tốt nghiệp", "công nhận tốt nghiệp"],
         "xét công nhận tốt nghiệp trung học cơ sở điều kiện 21 tuổi hạnh kiểm học lực lớp 9"),

        # Nghỉ hè giáo viên
        (["nghỉ hè", "nghỉ hè giáo viên"],
         "thời gian nghỉ hè hằng năm giáo viên cơ sở giáo dục phổ thông 08 tuần nghỉ phép Nghị định 84/2020"),

        # Miễn giảm học phí
        (["dân tộc thiểu số", "miễn giảm học phí", "miễn học phí"],
         "đối tượng miễn học phí dân tộc thiểu số vùng khó khăn"),

        # Thi bù / kiểm tra bù
        (["thi bù", "kiểm tra bù", "bị ốm không thi", "không tham gia thi",
          "không tham gia kiểm tra"],
         "học sinh không tham gia kiểm tra đánh giá bất khả kháng được kiểm tra đánh giá bù"),

        # Bổ nhiệm hạng giáo viên → Thông tư 01/2021 + 08/2023
        (["hạng ii", "hạng iii", "bổ nhiệm hạng", "thăng hạng", "chức danh nghề nghiệp"],
         "bổ nhiệm xếp lương viên chức giáo viên hạng chức danh Thông tư 01/2021 08/2023 sửa đổi"),

        # Ban đại diện cha mẹ học sinh
        (["quỹ lớp", "ban đại diện", "cha mẹ học sinh", "phụ huynh thu tiền"],
         "Ban đại diện cha mẹ học sinh Điều lệ Ban đại diện"),

        # Đánh giá học sinh THCS THPT
        (["đánh giá học sinh", "điểm số", "xếp loại học lực"],
         "đánh giá xếp loại kết quả học tập rèn luyện học sinh"),

        # Xử phạt vi phạm tuyển sinh
        (["xử phạt", "vi phạm tuyển sinh", "phạt tiền giáo dục", "tổ chức tuyển sinh"],
         "vi phạm quy định về tổ chức tuyển sinh phạt tiền thông báo tuyển sinh đề án"),
        (["tuyển sinh vượt chỉ tiêu", "vượt chỉ tiêu"],
         "tuyển sinh vượt chỉ tiêu trung học phổ thông phạt tiền mức phạt"),

        # Thăng hạng GV tiểu học → TT 02/2021 + 08/2023
        (["giáo viên tiểu học thăng hạng", "thăng hạng tiểu học"],
         "tiêu chuẩn chức danh nghề nghiệp giáo viên tiểu học hạng III hạng II mã số V.07.03 Thông tư 02/2021 08/2023"),

        # Thành lập trường mầm non → Luật 43/2019
        (["thành lập trường"],
         "điều kiện thành lập trường mầm non đề án quy hoạch cơ sở vật chất Luật Giáo dục 43/2019"),

        # Hệ thống giáo dục quốc dân
        (["hệ thống giáo dục", "cấp học", "trình độ đào tạo", "giáo dục quốc dân"],
         "hệ thống giáo dục quốc dân cấp học trình độ đào tạo mầm non tiểu học"),

        # Khung học phí
        (["học phí", "khung học phí", "nguyên tắc học phí"],
         "khung học phí nguyên tắc xác định bù đắp chi phí mức trần UBND"),

        # Đánh giá thường xuyên / định kỳ → Thông tư 22/2021
        (["đánh giá thường xuyên", "đánh giá định kỳ", "giữa kỳ", "cuối kỳ", "bài kiểm tra", "kiểm tra giữa kỳ", "kiểm tra định kỳ"],
         "đánh giá thường xuyên đánh giá định kỳ giữa kỳ cuối kỳ điểm số kiểm tra đánh giá xếp loại học sinh trung học cơ sở trung học phổ thông Thông tư 22/2021/TT-BGDĐT"),

        # Lên lớp, ở lại lớp, nghỉ học nhiều → Thông tư 22/2021
        (["nghỉ học", "tai nạn", "nghỉ quá 45 buổi", "lên lớp", "ở lại lớp", "nghỉ học nhiều", "cho phép lên lớp"],
         "học sinh nghỉ học vượt quá 45 buổi học trong năm học lên lớp ở lại lớp đánh giá kết quả học tập rèn luyện Thông tư 22/2021/TT-BGDĐT"),


        # Học bổng chính sách / cử tuyển → NĐ 84/2020
        (["học bổng", "dự bị đại học", "dân tộc nội trú", "cử tuyển"],
         "học bổng chính sách cử tuyển bồi hoàn dự bị đại học phổ thông dân tộc nội trú Nghị định 84/2020"),

        # Mục tiêu giáo dục
        (["mục tiêu giáo dục", "nguyên lý giáo dục", "tính chất giáo dục"],
         "mục tiêu giáo dục phát triển toàn diện đạo đức tri thức nhân dân dân tộc"),

        # ═══════════════════════════════════════════════════════════
        # NEW RULES — Targeting benchmark weak areas
        # ═══════════════════════════════════════════════════════════

        # Nâng chuẩn giáo viên → NĐ 71/2020
        (["nâng chuẩn", "nâng trình độ chuẩn", "lộ trình nâng chuẩn", "trình độ chuẩn được đào tạo",
          "bằng cử nhân giáo viên", "đào tạo cử nhân giáo viên"],
         "nâng trình độ chuẩn được đào tạo giáo viên lộ trình giai đoạn Nghị định 71/2020"),

        # Chương trình GDPT 2018 → TT 32/2018
        (["chương trình giáo dục phổ thông 2018", "chương trình mới", "sách giáo khoa mới",
          "chương trình 2018", "chương trình phổ thông mới", "lộ trình sgk"],
         "chương trình giáo dục phổ thông 2018 lộ trình áp dụng ngoại ngữ Thông tư 32/2018 Quyết định 16/2006"),

        # Giáo dục đại học / Chuyển đổi → Luật 34/2018 + NĐ 99/2019
        (["giáo dục đại học", "đại học tư thục", "không vì lợi nhuận", "chuyển đổi đại học",
          "cơ sở giáo dục đại học", "chuyển trường đại học thành đại học"],
         "cơ sở giáo dục đại học tư thục không vì lợi nhuận chuyển đổi Luật 34/2018 sửa đổi Nghị định 99/2019"),

        # Sinh viên sư phạm → NĐ 116/2020
        (["sinh viên sư phạm", "chính sách sư phạm", "hỗ trợ sư phạm", "sinh hoạt phí sư phạm",
          "bồi hoàn sư phạm", "bảo lưu sư phạm"],
         "sinh viên sư phạm chính sách hỗ trợ sinh hoạt phí bồi hoàn thôi học bảo lưu Nghị định 116/2020"),

        # Du học → NĐ 86/2021
        (["du học", "du học sinh", "học bổng ngân sách", "cử đi học nước ngoài",
          "lưu ban du học", "gia hạn du học", "tư vấn du học"],
         "du học sinh học bổng ngân sách nhà nước gia hạn lưu ban chuyển ngành Nghị định 86/2021"),

        # Phát triển sớm trí tuệ, học vượt lớp → Luật 43/2019
        (["phát triển sớm", "học vượt lớp", "vượt lớp"],
         "phát triển sớm về trí tuệ học vượt lớp tuổi vào học Luật Giáo dục 43/2019 Điều 28"),

        # Quyền nhà giáo, thỉnh giảng → Luật 43/2019
        (["thỉnh giảng", "quyền nhà giáo", "quyền giáo viên", "hợp đồng thỉnh giảng"],
         "nhà giáo quyền hợp đồng thỉnh giảng nghiên cứu khoa học Luật Giáo dục 43/2019 Điều 70"),

        # Ép buộc học thêm → Luật 43/2019
        (["ép buộc học thêm", "ép học sinh học thêm", "dạy thêm thu tiền"],
         "ép buộc học sinh học thêm thu tiền hành vi bị nghiêm cấm Luật Giáo dục 43/2019 Điều 22"),

        # Hỗ trợ mầm non khu công nghiệp → NĐ 105/2020 + Luật 43/2019
        (["khu công nghiệp", "mầm non khu công nghiệp", "hỗ trợ cơ sở vật chất mầm non",
          "con công nhân"],
         "cơ sở mầm non khu công nghiệp hỗ trợ trang bị cơ sở vật chất Nghị định 105/2020 Luật 43/2019"),

        # Giáo dục bắt buộc, tiểu học tư thục → Luật 43/2019
        (["giáo dục bắt buộc", "tiểu học tư thục", "hỗ trợ học phí tiểu học",
          "không đủ trường công lập"],
         "giáo dục tiểu học bắt buộc trường tư thục hỗ trợ tiền đóng học phí Luật Giáo dục 43/2019 Điều 99"),

        # Tài sản trường tư thục, vốn góp → Luật 43/2019
        (["tài sản trường", "sở hữu trường tư thục", "vốn góp nhà đầu tư"],
         "tài sản trường tư thục sở hữu nhà đầu tư vốn góp Luật Giáo dục 43/2019 Điều 102"),

        # Loại hình cơ sở giáo dục mầm non → Luật 43/2019
        (["nhà trẻ", "mẫu giáo", "trường mầm non", "lớp mầm non độc lập"],
         "cơ sở giáo dục mầm non nhà trẻ mẫu giáo trường mầm non 03 tháng tuổi Luật Giáo dục 43/2019 Điều 26"),

        # Hoàn thành tiểu học, bằng tốt nghiệp → Luật 43/2019
        (["hoàn thành tiểu học", "bằng tốt nghiệp tiểu học", "xác nhận học bạ"],
         "hoàn thành chương trình tiểu học hiệu trưởng xác nhận học bạ Luật Giáo dục 43/2019 Điều 34"),

        # Giảng viên ra nước ngoài nghiên cứu → NĐ 86/2021
        (["giảng viên nước ngoài", "nghiên cứu khoa học nước ngoài", "trao đổi học thuật"],
         "giảng viên giảng dạy nghiên cứu khoa học trao đổi học thuật nước ngoài Nghị định 86/2021 Điều 19"),

        # Thời gian giữ hạng, mã số V.07 → TT 01/2021 + TT 08/2023
        (["thời gian giữ hạng", "mã số v.07", "chuyển xếp", "tương đương hạng"],
         "thời gian giữ chức danh nghề nghiệp tương đương mã số V.07 Thông tư 01/2021 08/2023 sửa đổi"),

        # Nghỉ hưu giáo viên, còn công tác → NĐ 71/2020
        (["còn công tác", "tuổi nghỉ hưu", "07 năm công tác", "84 tháng"],
         "giáo viên còn đủ 07 năm công tác nghỉ hưu nâng trình độ chuẩn Nghị định 71/2020 Điều 2"),

        # Nhà đầu tư, rút vốn, lợi tức → Luật 34/2018
        (["nhà đầu tư", "rút vốn", "lợi tức", "cổ đông đại học"],
         "nhà đầu tư rút vốn lợi tức cơ sở giáo dục đại học tư thục Luật 34/2018 sửa đổi bổ sung Điều 7"),
    ]

    def expand(self, query: str) -> list:
        """Trả về list các query mở rộng (không bao gồm query gốc)."""
        q = query.lower()
        return [f"{query} {exp}" for kws, exp in self.EXPANSION_RULES if any(kw in q for kw in kws)]

    def get_target_docs(self, query: str) -> list:
        """Trả về list các so_hieu văn bản đích dựa trên keyword đặc trưng."""
        q = query.lower()
        targets = []
        
        # 1. Nâng chuẩn GV
        if any(kw in q for kw in ["nâng chuẩn", "trình độ chuẩn", "bằng cử nhân giáo viên"]):
            targets.append("71/2020/NĐ-CP")
            
        # 2. Thăng bổ nhiệm hạng GV
        if any(kw in q for kw in ["bổ nhiệm hạng", "thăng hạng", "hạng ii", "hạng iii"]):
            if "mầm non" in q:
                targets.append("01/2021/TT-BGDĐT")
                targets.append("08/2023/TT-BGDĐT")
            elif "tiểu học" in q:
                targets.append("02/2021/TT-BGDĐT")
                targets.append("08/2023/TT-BGDĐT")
            else:
                targets.append("01/2021/TT-BGDĐT")
                targets.append("08/2023/TT-BGDĐT")
                
        # 3. Đánh giá kiểm tra định kỳ học sinh THCS/THPT
        if any(kw in q for kw in ["giữa kỳ", "cuối kỳ", "bài kiểm tra", "kiểm tra định kỳ", 
                                  "kiểm tra thường xuyên", "nghỉ học nhiều", "ở lại lớp", 
                                  "lên lớp", "tai nạn", "kiểm tra giữa kỳ", "nghỉ học quá 45 buổi"]):
            targets.append("22/2021/TT-BGDĐT")
            
        # 4. Chuyển đổi loại hình đại học tư thục / công lập
        if "chuyển đổi" in q and "đại học" in q:
            targets.append("34/2018/QH14")
            targets.append("99/2019/NĐ-CP")
        elif any(kw in q for kw in ["đại học tư thục", "không vì lợi nhuận", "hội đồng trường", "đại học y"]):
            targets.append("34/2018/QH14")
            targets.append("99/2019/NĐ-CP")
            
        # 5. Thi tốt nghiệp THPT mới 2026
        if any(kw in q for kw in ["thi tốt nghiệp", "tốt nghiệp thpt", "2026", "24/2024"]):
            targets.append("24/2024/TT-BGDĐT")
            
        # 6. Tuyển sinh đại học 2022
        if any(kw in q for kw in ["tuyển sinh năm 2022", "tuyển sinh 2022", "tuyển sinh đại học"]):
            targets.append("08/2022/TT-BGDĐT")
            
        # 7. Học bổng chính sách / cử tuyển
        if any(kw in q for kw in ["học bổng", "cử tuyển", "dự bị đại học", "nội trú"]):
            targets.append("84/2020/NĐ-CP")
            
        # 8. Du học sinh / Nghiên cứu nước ngoài
        if any(kw in q for kw in ["du học", "nghiên cứu khoa học nước ngoài", "gia hạn du học", "sinh viên cử đi học", "giảng viên đại học cử ra nước ngoài"]):
            targets.append("86/2021/NĐ-CP")
            
        # 9. Sinh viên sư phạm hỗ trợ kinh phí
        if any(kw in q for kw in ["sinh viên sư phạm", "hỗ trợ sinh hoạt phí", "bồi hoàn", "116/2020", "bảo lưu sư phạm"]):
            targets.append("116/2020/NĐ-CP")
            
        # 10. Chuyển trường khác tỉnh
        if any(kw in q for kw in ["chuyển trường khác tỉnh", "giấy giới thiệu", "thủ tục chuyển trường"]):
            targets.append("43/2019/QH14")
            
        # 11. Chương trình giáo dục phổ thông 2018 / dạy ngoại ngữ
        if any(kw in q for kw in ["ngoại ngữ", "dạy ngoại ngữ", "ngoại ngữ 1", "chương trình giáo dục phổ thông 2018", "chương trình 2018"]):
            targets.append("32/2018/TT-BGDĐT")
            
        return list(dict.fromkeys(targets))  # Deduplicate while preserving order

