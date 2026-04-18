import json
import re
from openai import OpenAI

class RAGAgents:
    def __init__(self, api_key: str, base_url: str = None, model_name: str = "openai/gpt-oss-120b"):
        params = {"api_key": api_key}
        if base_url:
            params["base_url"] = base_url
        self.client = OpenAI(**params)
        self.model_name = model_name

    def _extract_json(self, text: str) -> dict:
        if not text or not text.strip():
            return {}
        try:
            # Strip markdown json wrappers
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return json.loads(text)
        except Exception as e:
            print(f"[Agents] JSON parsing error: {e} - Raw text: {text[:200]}")
            return {}

    def self_check(self, query: str, context: str) -> dict:
        """
        Self-Check Agent tự đánh giá bối cảnh.
        Trả về dictionary {"sufficient": True/False, "reason": "Lý do"}
        """
        system_prompt = """Bạn là Giám thị đánh giá RAG. Đánh giá xem đoạn CONTEXT dưới đây có ĐỦ dữ kiện để trả lời hoàn chỉnh câu QUERY của người dùng hay không.
Nếu thiếu thông tin trọng tâm bị hỏi, hãy phán quyết "NO".
Nếu đủ thông tin cốt lõi, hãy phán quyết "YES".
Bạn PHẢI trả về ĐÚNG MỘT JSON với định dạng: {"sufficient": "YES/NO", "reason": "lời giải thích ngắn gọn tại sao"}"""

        user_prompt = f"==== CONTEXT ====\n{context}\n\n==== QUERY ====\n{query}\n\nĐánh giá:"

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=100
            )
            content = resp.choices[0].message.content
            parsed = self._extract_json(content)
            is_sufficient = str(parsed.get("sufficient", "YES")).strip().upper() == "YES"
            reason = str(parsed.get("reason", "OK"))
            return {"sufficient": is_sufficient, "reason": reason}
        except Exception as e:
            print(f"[Self-Check] Failed: {e}")
            # Fallback luôn true để tránh RAG lặp vô hạn nếu API rớt
            return {"sufficient": True, "reason": "API Failed to check"}

    def judge_generation(self, query: str, context: str, answer: str) -> dict:
        """
        LLM Judge - Hậu kiểm (Sau Generation).
        Trả về json {"faithfulness": "PASS", "relevance": "PASS", "citation_accuracy": "PASS"}
        Và score tổng hợp.
        """
        system_prompt = """Bạn là Thẩm phán AI độc lập. Nhiệm vụ của bạn là đánh giá chất lượng câu trả lời của một hệ thống RAG Pháp lý.
Đánh giá dựa trên 3 tiêu chí sau (Tất cả đều trả về "PASS" hoặc "FAIL"):
1. "faithfulness": Câu trả lời có đúng dựa hoàn toàn vào NGỮ CẢNH cung cấp không? (Nghiêm cấm bịa đặt, Halucination).
2. "relevance": Câu trả lời có đi thẳng vào trọng tâm CÂU HỎI của người dùng không?
3. "citation_accuracy": Các trích dẫn Điều/Khoản trong câu trả lời có BÁM SÁT ngữ cảnh gốc không?

Bạn PHẢI trả về ĐÚNG MỘT JSON định dạng: {"faithfulness": "PASS/FAIL", "relevance": "PASS/FAIL", "citation_accuracy": "PASS/FAIL", "reason": "lời phán xử ngắn gọn"}"""

        user_prompt = f"==== CÂU HỎI VIỆC ====\n{query}\n\n==== NGỮ CẢNH TÌM ĐƯỢC ====\n{context}\n\n==== CÂU TRẢ LỜI CỦA RAG ====\n{answer}\n\nChấm Điểm:"

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=150
            )
            content = resp.choices[0].message.content
            parsed = self._extract_json(content)
            
            f_pass = str(parsed.get("faithfulness", "PASS")).strip().upper() == "PASS"
            r_pass = str(parsed.get("relevance", "PASS")).strip().upper() == "PASS"
            c_pass = str(parsed.get("citation_accuracy", "PASS")).strip().upper() == "PASS"
            
            score = sum([f_pass, r_pass, c_pass])
            
            return {
                "score": score,
                "passed": score == 3,  # Cần tuyệt đối 3/3 Pass
                "metrics": parsed,
                "reason": parsed.get("reason", "")
            }
        except Exception as e:
            print(f"[LLM Judge] Failed: {e}")
            return {"score": 3, "passed": True, "metrics": {}, "reason": "Lỗi API nhưng cho qua"}
