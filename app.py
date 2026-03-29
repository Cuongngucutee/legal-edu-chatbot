import os
import streamlit as st
import time

# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from book_index import BookIndex
from retriever import BookRAGRetriever
from generator import BookRAGGenerator

st.set_page_config(page_title="BookRAG Pháp Luật Demo", layout="wide")
st.title("📚 BookRAG: Hệ Thống Truy Vấn Pháp Luật")
st.markdown("Sử dụng cấu trúc Hierarchical Tree (JSON) & Knowledge Graph kết hợp với **Vietnamese_Law_Embedding** và **Gemini**.")

# Setup config
with st.sidebar:
    st.header("Cấu Hình")
    # link : https://api.groq.com/openai/v1
    api_key = st.text_input("OpenAI/OpenRouter API Key", type="password")
    base_url = st.text_input("Base URL (Để trống hoặc dùng https://openrouter.ai/api/v1)", value="")
    model_name = st.text_input("Model Name", value="openai/gpt-oss-120b")
    st.markdown("- **Embedding**: `namnguyenba2003/Vietnamese_Law_Embedding_finetuned_v3_256dims`")
    
    st.markdown("---")
    st.markdown("**Quá trình khởi tạo BookIndex**")
    
    # Initialize components
    if "index" not in st.session_state:
        st.session_state.index_status = "Đang tải mô hình & dữ liệu..."
        
@st.cache_resource
def load_index():
    data_dir = "data/final"
    kg_path = "outputs/knowledge_graph/entity_graph.json"
    index = BookIndex(data_dir, kg_path)
    index.load_index()
    return index

with st.spinner("Đang xây dựng BookIndex (Tree + Knowledge Graph)... Vui lòng đợi (Lần đầu sẽ mất thời gian load model embedding)"):
    index = load_index()
    retriever = BookRAGRetriever(index)

st.sidebar.success("✅ BookIndex & Retriever Đã Sẵn Sàng!")
st.sidebar.info(f"Số lượng Node: {index.graph.number_of_nodes()}\nSố lượng Edge: {index.graph.number_of_edges()}")

# Main Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hãy đặt câu hỏi pháp lý..."):
    if not api_key:
        st.warning("Vui lòng nhập API Key ở Sidebar!")
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Đang phân tích BookIndex để tìm ngữ cảnh...", expanded=True) as status:
            start_time = time.time()
            st.write("1. Nhúng câu hỏi với `namnguyenba2003/Vietnamese_Law_Embedding_finetuned_v3_256dims`")
            st.write("2. Tìm kiếm các Node đầu vào từ Vector Index...")
            
            # Retrieval
            context = retriever.retrieve(prompt, top_k=3, max_hops=1)
            
            st.write("3. Duyệt Đồ Thị (Graph Traversal & Tree Hierarchy) để thiết lập ngữ cảnh hoàn chỉnh.")
            
            st.markdown("**Ngữ Cảnh Tìm Thấy:**")
            with st.expander("Bấm để xem chi tiết Ngữ Cảnh Dành Cho LLM", expanded=False):
                st.text(context)
                
            status.update(label=f"Đã tìm xong ngữ cảnh ({time.time() - start_time:.2f}s)! Đang sinh câu trả lời...", state="running")
            
            # Generation
            generator = BookRAGGenerator(
                api_key=api_key, 
                base_url=base_url if base_url.strip() else None, 
                model_name=model_name, 
                retriever=retriever
            )
            try:
                answer = generator.generate_answer(prompt, context=context)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                status.update(label="Hoàn thành!", state="complete")
            except Exception as e:
                st.error(f"Lỗi khi gọi API LLM: {e}")
                status.update(label="Lỗi trong quá trình sinh", state="error")
