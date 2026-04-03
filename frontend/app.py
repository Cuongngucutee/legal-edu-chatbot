import os
import streamlit as st
import time
from dotenv import load_dotenv

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

# Add backend/core to path
import sys
sys.path.append(os.path.join(ROOT_DIR, "backend", "core"))

from book_index import BookIndex
from retriever import BookRAGRetriever
from generator import BookRAGGenerator
from agents import RAGAgents
from rule_engine import RuleEngine
import json

st.set_page_config(page_title="BookRAG Pháp Luật Demo", layout="wide")
st.title("📚 BookRAG: Hệ Thống Truy Vấn Pháp Luật")
st.markdown("Sử dụng cấu trúc Hierarchical Tree (JSON) & Knowledge Graph kết hợp với **Vietnamese_Law_Embedding** và **Gemini**.")

# Setup config
with st.sidebar:
    st.header("Cấu Hình")
    # link : https://api.groq.com/openai/v1
    api_key = st.text_input("OpenAI/OpenRouter API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    base_url = st.text_input("Base URL (Để trống hoặc dùng https://openrouter.ai/api/v1)", value=os.getenv("OPENAI_BASE_URL", ""))
    model_name = st.text_input("Model Name", value=os.getenv("OPENAI_MODEL_NAME", "openai/gpt-oss-120b"))
    st.markdown("- **Embedding**: `namnguyenba2003/Vietnamese_Law_Embedding_finetuned_v3_256dims`")
    
    st.markdown("---")
    st.markdown("**Quá trình khởi tạo BookIndex**")
    
    # Initialize components
    if "index" not in st.session_state:
        st.session_state.index_status = "Đang tải mô hình & dữ liệu..."
        
@st.cache_resource
def load_index():
    data_dir = os.path.join(ROOT_DIR, "data", "final")
    kg_path = os.path.join(ROOT_DIR, "outputs", "knowledge_graph", "entity_graph.json")
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
        
    # RULE ENGINE GATE CONSTRUCT
    rule_engine = RuleEngine(book_index=index)
    val_res = rule_engine.validate(prompt)
    if not val_res["pass"]:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": f"🚨 **Từ chối (Rule Engine):** {val_res['reason']}"})
        st.rerun()
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Đang phân tích ý định câu hỏi (Query Understanding)...", expanded=True) as status:
            start_time = time.time()
            # 1. Tầng 1: Query Classification
            agent = RAGAgents(
                api_key=api_key, 
                base_url=base_url if base_url.strip() else None, 
                model_name=model_name
            )
            query_type = agent.classify_query(prompt)
            st.write(f"🕵️ **Tầng 1 (Query Classifier):** Phát hiện loại câu hỏi `[{query_type.upper()}]`")
            
            # 2. Tầng 2: Adaptive Hybrid Retrieval + Cross-Encoder Reranking
            st.write(f"🔍 **Tầng 2 (Adaptive Hybrid Retrieval):** BM25 + FAISS + Graph Search...")
            base_top_k = 5
            base_max_hops = 1
            context = retriever.retrieve(prompt, query_type=query_type, top_k=base_top_k, max_hops=base_max_hops)
            st.write(f"✨ Đã áp dụng Lọc chéo (Cross-Encoder Reranker) giữ lại các Node tốt nhất.")
            
            # 3. Tầng 3: Self-Check Agent
            st.write(f"⚖️ **Tầng 3 (Self-Check Agent):** Đánh giá chất lượng ngữ cảnh...")
            check_result = agent.self_check(prompt, context)
            if check_result["sufficient"]:
                st.write(f"✅ **Self-Check Pass:** Ngữ cảnh đầy đủ ({check_result['reason']})")
            else:
                st.write(f"⚠️ **Self-Check Fail:** Thiếu hụt thông tin ({check_result['reason']})")
                st.write(f"🔄 Kích hoạt trích xuất sâu hơn (max_hops=2, top_k=10)...")
                # Retry logic
                context = retriever.retrieve(prompt, query_type="tong_hop", top_k=10, max_hops=2)
                st.write(f"✅ Đã thu thập thêm dữ kiện vòng 2.")
            
            st.markdown("**Ngữ Cảnh Tìm Thấy:**")
            with st.expander("Bấm để xem chi tiết Ngữ Cảnh Dành Cho LLM", expanded=False):
                st.text(context)
                
            status.update(label=f"Đã duyệt xong quy trình trong ({time.time() - start_time:.2f}s)! Đang sinh câu trả lời...", state="running")
            
            # 4. Tầng 4: Structured Generation + LLM Judge
            generator = BookRAGGenerator(
                api_key=api_key, 
                base_url=base_url if base_url.strip() else None, 
                model_name=model_name, 
                retriever=retriever
            )
            try:
                answer = generator.generate_answer(prompt, context=context)
                
                status.update(label="Đang đánh giá kết quả (LLM Judge Eval)...", state="running")
                judge_res = agent.judge_generation(prompt, context, answer)
                
                # Setup offline logging
                outputs_dir = os.path.join(ROOT_DIR, "outputs")
                os.makedirs(outputs_dir, exist_ok=True)
                log_data = {"query": prompt, "answer": answer, "eval": judge_res}
                try:
                    with open(os.path.join(outputs_dir, "offline_eval_log.jsonl"), "a", encoding="utf-8") as lf:
                        lf.write(json.dumps(log_data, ensure_ascii=False) + "\n")
                except:
                    pass
                
                if judge_res["passed"]:
                    st.write("🛂 **LLM Judge:** Hoàn hảo (Đạt 3/3 Tiêu chí). Cho phép hiển thị.")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    status.update(label="Hoàn thành!", state="complete")
                else:
                    fail_msg = f"Tôi đã tổng hợp dữ liệu nhưng kết quả bị từ chối bởi AI Judge (Điểm: {judge_res['score']}/3). Lư do: {judge_res['reason']}. Vui lòng thử cách hỏi từ khóa rõ ràng hơn!"
                    st.error("🛂 **LLM Judge Cảnh Báo:** " + judge_res['reason'])
                    st.markdown(fail_msg)
                    st.session_state.messages.append({"role": "assistant", "content": fail_msg})
                    status.update(label="Bị chặn bởi LLM Judge", state="error")
            except Exception as e:
                st.error(f"Lỗi khi gọi API LLM: {e}")
                status.update(label="Lỗi trong quá trình sinh", state="error")
