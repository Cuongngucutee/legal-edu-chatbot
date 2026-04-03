import streamlit as st
import time
import os

from src.indexing import build_pipeline
from src.models import PipelineAnswer

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="BookRAG Pháp Luật",
    page_icon="⚖️",
    layout="wide",
)

# ── Khởi tạo Pipeline ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_bookrag_pipeline():
    """Tạo cached pipeline để không phải build lại sau mỗi lần reload."""
    with st.spinner("Đang tải dữ liệu và xây dựng BookRAG Pipeline (Mất khoảng 30s lần đầu)..."):
        # Import settings and build pipeline
        from src.indexing import build_pipeline
        pipeline = build_pipeline(verbose=False)
    return pipeline

try:
    pipeline = load_bookrag_pipeline()
except Exception as e:
    st.error(f"Lỗi khởi tạo pipeline: {e}")
    st.stop()


# ── CSS Tùy Chỉnh ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Styling cho expander context */
    .context-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        border-left: 4px solid #4CAF50;
    }
    .context-title {
        font-weight: bold;
        color: #1f2937;
        margin-bottom: 5px;
    }
    .context-text {
        color: #4b5563;
        font-size: 0.9em;
    }
    /* Citation styling */
    .citation {
        background-color: #e0f2fe;
        padding: 2px 5px;
        border-radius: 4px;
        color: #0369a1;
        font-size: 0.85em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ── Giao diện ─────────────────────────────────────────────────
st.title("⚖️ Trợ Lý Pháp Luật Giáo Dục (BookRAG)")
st.caption("Hệ thống hỏi đáp ứng dụng Retrieval-Augmented Generation (RAG) phân cấp và Đồ thị tri thức")

# ── Thanh bên (Sidebar) ───────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cấu hình")
    st.markdown("**Trạng thái hệ thống:**")
    stats = pipeline.get_stats()
    st.success(f"✅ Đã tải {stats.get('total_chunks', 0)} chunks từ {stats.get('tree_nodes', 0)} node.")
    st.info(f"🧠 Knowledge Graph: {stats.get('kg_nodes', 0)} nodes, {stats.get('kg_edges', 0)} edges.")
    
    st.divider()
    st.markdown("### Tùy chọn hiển thị")
    show_reasoning = st.checkbox("Hiển thị chuỗi suy luận (CoT)", value=True)
    show_context = st.checkbox("Hiển thị văn bản (Context)", value=True)
    
    st.divider()
    st.markdown("### Ví dụ câu hỏi")
    st.markdown("""
    - *Giáo dục chính quy là gì?*
    - *Điều kiện thành lập trường đại học?*
    - *So sánh quyền tự chủ đại học giữa Luật 08/2012 và Luật 34/2018.*
    - *Thẩm quyền công nhận hiệu trưởng trường đại học dân lập là của ai?*
    """)
    if st.button("Làm mới đoạn chat"):
        st.session_state.messages = []
        st.rerun()

# ── Khởi tạo lưu trữ hội thoại ────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Hiển thị tin nhắn cũ ─────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Nếu là câu trả lời của bot, hiển thị thêm các phần ẩn
        if message["role"] == "assistant" and "raw_answer" in message:
            ans: PipelineAnswer = message["raw_answer"]
            
            if show_reasoning and ans.reasoning_trace:
                with st.expander("🤔 Các bước suy luận (CoT)", expanded=False):
                    st.text(ans.reasoning_trace)
                    
            if show_context and ans.retrieved_chunks:
                with st.expander(f"📄 Ngữ cảnh tìm thấy ({len(ans.retrieved_chunks)} điều khoản)", expanded=False):
                    for rc in ans.retrieved_chunks:
                        st.markdown(f"""
                        <div class="context-box">
                            <div class="context-title">[{rc.chunk.source}] {rc.chunk.article_title} (Score: {rc.score:.2f})</div>
                            <div class="context-text">{rc.chunk.text}</div>
                        </div>
                        """, unsafe_allow_html=True)

# ── Nhận câu hỏi mới ─────────────────────────────────────────
if prompt := st.chat_input("Hãy nhập câu hỏi pháp luật (Ví dụ: Thẩm quyền mở ngành đào tạo?)"):
    # Hiển thị tin nhắn của người dùng
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Hiển thị tin nhắn của hệ thống & tiến trình xử lý
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.status("Đang tra cứu cơ sở dữ liệu luật...", expanded=True) as status:
            t0 = time.time()
            st.write("🔍 Phân tích intent và entity...")
            
            # Xử lý RAG
            answer: PipelineAnswer = pipeline.query_raw(prompt, top_k=8)
            
            st.write(f"📚 Đã tìm thấy {len(answer.retrieved_chunks)} điều khoản liên quan.")
            st.write("💭 Đang thiết lập lập luận logic...")
            
            t_total = time.time() - t0
            status.update(label=f"Hoàn thành trong {t_total:.2f}s!", state="complete", expanded=False)

        # Hiển thị câu trả lời (Markdown)
        message_placeholder.markdown(answer.answer)
        
        # Thêm expandable block ngay dưới câu trả lời
        if show_reasoning and answer.reasoning_trace:
            with st.expander("🤔 Các bước suy luận (CoT)", expanded=False):
                st.text(answer.reasoning_trace)
                
        if show_context and answer.retrieved_chunks:
            with st.expander(f"📄 Ngữ cảnh tìm thấy ({len(answer.retrieved_chunks)} điều khoản)", expanded=False):
                for rc in answer.retrieved_chunks:
                    st.markdown(f"""
                    <div class="context-box">
                        <div class="context-title">[{rc.chunk.source}] {rc.chunk.article_title} (Score: {rc.score:.2f})</div>
                        <div class="context-text">{rc.chunk.text}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
        # Lưu vào trạng thái
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer.answer,
            "raw_answer": answer
        })
