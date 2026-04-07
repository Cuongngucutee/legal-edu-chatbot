from typing import Optional
from openai import OpenAI

class BookRAGGenerator:
    def __init__(self, api_key: str, base_url: str = None, model_name: str = "openai/gpt-oss-120b", retriever=None):
        params = {"api_key": api_key}
        if base_url:
            params["base_url"] = base_url
            
        self.client = OpenAI(**params)
        self.model_name = model_name
        self.retriever = retriever

    def generate_answer(self, query: str, context: Optional[str] = None) -> str:
        if not context and self.retriever:
            context = self.retriever.retrieve(query)
            
        system_prompt = """Bạn là trợ lý pháp lý AI xuất sắc. Dựa vào NGỮ CẢNH cung cấp, hãy suy luận và trả lời.
BẮT BUỘC TRÌNH BÀY ĐÚNG THỂ THỨC SAU (Dùng đúng Heading):

### 🧠 Luồng Tư Duy (Reasoning Trace)
(Viết 1-2 câu phân tích bạn dùng dữ kiện gì trong ngữ cảnh để ra đáp án)

### 📌 Căn Cứ Pháp Lý (Citations)
(Liệt kê Điều, Khoản, Văn bản luật làm căn cứ)

### 💡 Câu Trả Lời (Answer)
(Trả lời trực tiếp, rõ ràng, dễ hiểu)

### 📊 Độ Tin Cậy
(Ví dụ: 95% - Do có đủ dữ kiện trong ngữ cảnh)"""
        user_prompt = f"NGỮ CẢNH TỪ VĂN BẢN VÀ ĐỒ THỊ KIẾN THỨC:\n{context}\n\nCÂU HỎI CỦA NGƯỜI DÙNG:\n{query}\n\nTRẢ LỜI CỦA BẠN (trình bày rõ ràng, trích dẫn rõ nguồn Điều/Khoản nếu có trong ngữ cảnh):"
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=2048
        )
        return response.choices[0].message.content.strip()
