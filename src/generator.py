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
            
        system_prompt = "Bạn là một trợ lý pháp lý AI xuất sắc. Dựa vào bộ dữ liệu ngữ cảnh (được truy xuất dựa trên phương pháp Node & Đồ thị BookRAG), hãy trả lời câu hỏi của người dùng một cách chính xác."
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
