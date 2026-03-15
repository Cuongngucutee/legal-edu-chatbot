import streamlit as st
import os
from groq import Groq
from dotenv import load_dotenv
from core_logic import VietnameseEmbedding, get_legal_answer
from ui_style import apply_custom_styles, sidebar_content

# 1. Cấu hình
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
KG_PATH = "outputs/knowledge_graph/knowledge_graph_full.json"

st.set_page_config(page_title="Trợ Lý Pháp Luật", layout="wide")
apply_custom_styles()
sidebar_content()

# 2. Khởi tạo Resources
@st.cache_resource
def load_data():
    client_groq = Groq(api_key=GROQ_API_KEY)
    return client_groq

client_groq = load_data()

st.title("⚖️ Trợ lý Luật sư AI - Hệ thống Pháp luật Việt Nam")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Ô nhập liệu
if prompt := st.chat_input("Hỏi tôi bất cứ điều gì về luật..."):
    # Lưu câu hỏi của user vào session state để hiển thị lịch sử chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    # ...
    with st.chat_message("assistant"):
        with st.spinner("Đang kết nối database và tìm kiếm..."):
            # CHÚ Ý: Truyền None vào chỗ collection vì core_logic giờ tự kết nối Qdrant
            result = get_legal_answer(prompt, None, client_groq, KG_PATH)
            
            # Unpack an toàn
            if isinstance(result, tuple) and len(result) == 3:
                answer, context, metas = result
            else:
                answer, context, metas = result, "", []
                
            st.markdown(answer)
            
            # Hiển thị nguồn trích dẫn
            if metas and len(metas) > 0:
                st.write("---")
                st.caption("Nguồn trích dẫn:")
                num_cols = min(len(metas), 4)
                cols = st.columns(num_cols)
                for i in range(num_cols):
                    m = metas[i]
                    source = m.get('source', 'Luật')
                    article = m.get('article_number', 'mới')
                    cols[i].info(f"📍 {source}\nĐiều {article}") 
            else:
                st.caption("Trả lời dựa trên kiến thức chung (Không tìm thấy nguồn trực tiếp trong DB).")
        
    st.session_state.messages.append({"role": "assistant", "content": answer})