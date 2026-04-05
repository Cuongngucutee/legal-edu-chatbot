import streamlit as st
import os
from groq import Groq
from dotenv import load_dotenv
from core_logic import VietnameseEmbedding, get_legal_answer
from ui_style import apply_custom_styles, sidebar_content

# 1. Cấu hình
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
KG_PATH = "outputs/knowledge_graph/legal_graph_final.json"

st.set_page_config(page_title="Trợ Lý Pháp Luật AI", layout="wide", page_icon="⚖️")
apply_custom_styles()
sidebar_content()

# 2. Khởi tạo Resources
@st.cache_resource
def load_data():
    client_groq = Groq(api_key=GROQ_API_KEY)
    return client_groq

client_groq = load_data()

st.title("⚖️ Trợ lý Luật sư AI - Hybrid GraphRAG")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Ô nhập liệu
if prompt := st.chat_input("Hỏi tôi về Luật Giáo dục, đối tượng áp dụng, hoặc các dẫn chiếu..."):
    # Lưu và hiển thị câu hỏi User
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🧠 Đang lục tìm trong kho luật..."):
            # Unpack thêm biến kg_info từ core_logic
            result = get_legal_answer(prompt, None, client_groq, KG_PATH)
            
            if isinstance(result, tuple) and len(result) == 4:
                answer, context, metas, kg_info = result
            else:
                answer, context, metas, kg_info = result, "", [], ""
            
            # 1. Hiển thị câu trả lời của AI
            st.markdown(answer)
            
            # 2. HIỂN THỊ CHI TIẾT CÁC ĐIỀU LUẬT ĐÃ LẤY RA
            st.write("---")
            with st.expander("🔍 Tra cứu căn cứ pháp lý & Graph liên quan", expanded=False):
                # Tạo 2 Tab chính
                tab1, tab2 = st.tabs(["📄 Nội dung chi tiết (Văn bản)", "🕸️ Thực thể & Mối liên hệ (Graph)"])
                
                with tab1:
                    if metas:
                        st.info("📌 Các điều luật được hệ thống trích xuất để trả lời:")
                        for m in metas:
                            # Lấy thông tin từ metadata của từng đoạn (point)
                            so_dieu = m.get('article_number', '...')
                            ten_luat = m.get('source', 'Văn bản pháp luật')
                            noi_dung = m.get('text', 'Không có nội dung')
                            
                            # Hiển thị tiêu đề theo format: Điều X - Luật Y
                            with st.container():
                                st.markdown(f"### 📑 Điều {so_dieu} | {ten_luat}")
                                # Dùng code hoặc markdown tùy mày, nhưng markdown nhìn sẽ tự nhiên hơn
                                st.write(noi_dung)
                                st.divider() # Dấu gạch ngang tách giữa các điều luật khác nhau
                    else:
                        st.warning("Không tìm thấy nội dung văn bản trực tiếp.")
                        
                with tab2:
                    if kg_info:
                        st.success("🕸️ Phân tích thực thể và dẫn chiếu từ Knowledge Graph:")
                        
                        # Tách từng dòng từ kg_info
                        lines = kg_info.split('\n')
                        for line in lines:
                            if "liên quan đến:" in line:
                                # Tách phần Điều và phần nội dung
                                parts = line.split("liên quan đến:", 1)
                                dieu_title = parts[0].strip()
                                content = parts[1].strip()
                                
                                # Kiểm tra xem có dẫn chiếu nội bộ [REF] không
                                ref_text = ""
                                if "[REF]" in content:
                                    content, ref_text = content.split("[REF]", 1)
                                
                                # --- HIỂN THỊ ---
                                with st.container():
                                    st.markdown(f"#### 📍 {dieu_title}")
                                    
                                    # Hiện thực thể liên quan
                                    st.write(f"🔹 **Thực thể:** {content.strip()}")
                                    
                                    # Nếu có dẫn chiếu thì hiện thêm dòng phụ
                                    if ref_text:
                                        st.caption(f"🔗 *Dẫn chiếu nội bộ đến các Điều: {ref_text.strip()}*")
                                    
                                    st.divider() # Vạch kẻ ngăn cách giữa các Điều
                    else:
                        st.warning("Không tìm thấy mối liên hệ thực thể cụ thể trong Graph.")

            # 3. Hiển thị nguồn trích dẫn (Các thẻ Info nhỏ bên dưới)
            if metas:
                st.caption("📍 Danh sách nguồn (Căn cứ pháp lý):")
                
                # --- BƯỚC 1: LỌC TRÙNG NGUỒN (CHỈ GIỮ LẠI CÁC ĐIỀU KHÁC NHAU) ---
                seen_sources = set()
                unique_metas = []
                for m in metas:
                    # Tạo một cái 'ID' kết hợp giữa tên Luật và Số điều
                    source_id = f"{m.get('source')}_{m.get('article_number')}"
                    if source_id not in seen_sources:
                        unique_metas.append(m)
                        seen_sources.add(source_id)
                
                # --- BƯỚC 2: HIỂN THỊ CÁC CỘT (Tối đa 4 nguồn cho đẹp) ---
                display_metas = unique_metas[:4] 
                cols = st.columns(len(display_metas))
                
                for i, m in enumerate(display_metas):
                    # Hiển thị từng thẻ nguồn
                    with cols[i]:
                        st.info(f"**{m.get('source', 'Luật')}**\n\nĐiều {m.get('article_number', '...')}")
        
    st.session_state.messages.append({"role": "assistant", "content": answer})