import os
# CRITICAL: đặt trước MỌI import — tránh segfault MPS khi load nhiều model
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import streamlit as st
import time
from dotenv import load_dotenv

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

# Add backend/core to path
import sys
sys.path.append(os.path.join(ROOT_DIR, "backend", "core"))

# NOTE: KHÔNG import book_index/retriever ở đây!
# SentenceTransformer init MPS sớm → Qwen safetensors sẽ SEGFAULT.
# Import lazy bên trong cached functions SAU khi Qwen đã load.
from query_intent import QwenIntentClassifier, QueryIntent, intent_to_legacy_type
import json

st.set_page_config(page_title="BookRAG Pháp Luật Demo", layout="wide")
st.title("📚 BookRAG: Hệ Thống Truy Vấn Pháp Luật")
st.markdown("Sử dụng cấu trúc Hierarchical Tree (JSON) & Knowledge Graph kết hợp với **Vietnamese_Law_Embedding** và **Gemini**.")

# Setup config
with st.sidebar:
    st.header("Cấu Hình LLM")
    
    provider = st.selectbox("Provider", ["OpenRouter", "Groq", "OpenAI"], index=0)
    
    # Auto-fill base_url và model gợi ý theo provider
    provider_config = {
        "OpenRouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "model_hint": "google/gemini-2.0-flash-001",
            "key_hint": "sk-or-v1-...",
        },
        "Groq": {
            "base_url": "https://api.groq.com/openai/v1",
            "model_hint": "llama3-70b-8192",
            "key_hint": "gsk_...",
        },
        "OpenAI": {
            "base_url": "",
            "model_hint": "gpt-4o-mini",
            "key_hint": "sk-...",
        },
    }
    cfg = provider_config[provider]
    
    api_key = st.text_input(
        "API Key", 
        type="password", 
        value=os.getenv("OPENAI_API_KEY", ""),
        placeholder=cfg["key_hint"],
    )
    base_url = st.text_input(
        "Base URL", 
        value=cfg["base_url"],
        disabled=(provider != "OpenAI"),  # Auto-fill, chỉ editable cho OpenAI
    )
    model_name = st.text_input(
        "Model Name", 
        value=os.getenv("OPENAI_MODEL_NAME", "") or cfg["model_hint"],
        placeholder=cfg["model_hint"],
    )
    st.markdown("- **Embedding**: `namnguyenba2003/Vietnamese_Law_Embedding_finetuned_v3_256dims`")
    
    st.markdown("---")
    st.markdown("**Quá trình khởi tạo BookIndex**")
    
    # Initialize components
    if "index" not in st.session_state:
        st.session_state.index_status = "Đang tải mô hình & dữ liệu..."

@st.cache_resource
def load_qwen_model():
    """Load Fine-tuned Qwen model TRƯỚC FAISS — tránh segfault do FAISS mmap xung đột safetensors.
    Nếu vẫn crash, đổi use_regex_only=True."""
    classifier = QwenIntentClassifier(
        model_name="manhcuong2005/qwen2.5-1.5b-legal-edu-v5",
        use_regex_only=False,  # True nếu Qwen vẫn crash
    )
    classifier._load_model()  # Eager load TRƯỚC FAISS
    return classifier

@st.cache_resource
def load_index():
    # Lazy import — phải sau khi Qwen đã load xong
    from book_index import BookIndex
    data_dir = os.path.join(ROOT_DIR, "data", "final")
    kg_path = os.path.join(ROOT_DIR, "outputs", "knowledge_graph", "entity_graph.json")
    index = BookIndex(data_dir, kg_path)
    index.load_index()
    return index

@st.cache_resource
def load_retriever(_index):
    """Load Retriever + CrossEncoder (cached, chỉ load 1 lần)."""
    from retriever import BookRAGRetriever
    return BookRAGRetriever(_index)

# ── CRITICAL LOAD ORDER: Qwen → BookIndex/FAISS → Retriever ──
with st.spinner("Đang tải Qwen Intent Classifier..."):
    intent_classifier = load_qwen_model()

with st.spinner("Đang xây dựng BookIndex (Tree + Knowledge Graph)... Vui lòng đợi"):
    index = load_index()
    # Gán doc_registry cho classifier sau khi index load
    intent_classifier.doc_registry = index.doc_registry
    retriever = load_retriever(index)

st.sidebar.success("✅ BookIndex & Retriever Đã Sẵn Sàng!")
st.sidebar.info(f"Số lượng Node: {index.graph.number_of_nodes()}\nSố lượng Edge: {index.graph.number_of_edges()}\nDoc Registry: {len(index.doc_registry)} entries → {len(index.doc_nodes)} documents")

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
        
    # RULE ENGINE GATE CONSTRUCT (Tạm thời tắt)
    # from rule_engine import RuleEngine
    # rule_engine = RuleEngine(book_index=index)
    # val_res = rule_engine.validate(prompt)
    # if not val_res["pass"]:
    #     st.session_state.messages.append({"role": "user", "content": prompt})
    #     st.session_state.messages.append({"role": "assistant", "content": f"🚨 **Từ chối (Rule Engine):** {val_res['reason']}"})
    #     st.rerun()
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Đang phân tích ý định câu hỏi (Query Understanding)...", expanded=True) as status:
            start_time = time.time()
            
            # ══════════════════════════════════════════════════════════════
            # TẦNG 1: Query Intent Classification
            # ══════════════════════════════════════════════════════════════
            intent = intent_classifier.analyze(prompt)
            
            classifier_label = "Regex Classifier" if intent._fallback_used else "Qwen2.5 Classifier"
            st.write(f"🧠 **Tầng 1 ({classifier_label}):** Phát hiện loại `[{intent.type.upper()}]` ({intent.total_time_ms:.0f}ms)")
            
            # Hiển thị chi tiết QueryIntent
            with st.expander("📋 Chi tiết QueryIntent JSON", expanded=False):
                st.json(intent.to_dict())
            
            if intent._fallback_used:
                st.write("⚠️ *Model Qwen không khả dụng — đã dùng Regex fallback*")
            
            if intent.documents:
                st.write(f"📄 Văn bản nhận diện: `{', '.join(intent.documents)}`")
            if intent.article_hint:
                st.write(f"📌 Gợi ý Điều: `Điều {intent.article_hint}`")
            
            # ══════════════════════════════════════════════════════════════
            # TẦNG 2: Adaptive Hybrid Retrieval (Intent-driven)
            # ══════════════════════════════════════════════════════════════
            st.write(f"🔍 **Tầng 2 (Intent-driven Retrieval):** Chiến lược `{intent.type}` — BM25 + FAISS + KG Scope...")
            base_top_k = 5
            context = retriever.retrieve(prompt, intent=intent, top_k=base_top_k)
            st.write(f"✨ Đã truy xuất {len(context)} ký tự ngữ cảnh.")
            
            # ══════════════════════════════════════════════════════════════
            # TẦNG 3: Self-Check Agent (API-based)
            # ══════════════════════════════════════════════════════════════
            from agents import RAGAgents
            agent = RAGAgents(
                api_key=api_key, 
                base_url=base_url if base_url.strip() else None, 
                model_name=model_name
            )
            
            st.write(f"⚖️ **Tầng 3 (Self-Check Agent):** Đánh giá chất lượng ngữ cảnh...")
            check_result = agent.self_check(prompt, context)
            if check_result["sufficient"]:
                st.write(f"✅ **Self-Check Pass:** Ngữ cảnh đầy đủ ({check_result['reason']})")
            else:
                st.write(f"⚠️ **Self-Check Fail:** Thiếu hụt thông tin ({check_result['reason']})")
                st.write(f"🔄 Kích hoạt trích xuất sâu hơn (global search, top_k=10)...")
                # Retry logic — fallback về cross_document search
                from query_intent import QueryIntent as QI
                retry_intent = QI(type="cross_document", keywords=intent.keywords, search_scope="global", topic=intent.topic)
                context = retriever.retrieve(prompt, intent=retry_intent, top_k=10)
                st.write(f"✅ Đã thu thập thêm dữ kiện vòng 2.")
            
            st.markdown("**Ngữ Cảnh Tìm Thấy:**")
            with st.expander("Bấm để xem chi tiết Ngữ Cảnh Dành Cho LLM", expanded=False):
                st.text(context)
                
            status.update(label=f"Đã duyệt xong quy trình trong ({time.time() - start_time:.2f}s)! Đang sinh câu trả lời...", state="running")
            
            # ══════════════════════════════════════════════════════════════
            # TẦNG 4: Structured Generation + LLM Judge
            # ══════════════════════════════════════════════════════════════
            from generator import BookRAGGenerator
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
                
                # Setup offline logging (thêm intent vào log)
                outputs_dir = os.path.join(ROOT_DIR, "outputs")
                os.makedirs(outputs_dir, exist_ok=True)
                log_data = {
                    "query": prompt,
                    "intent": intent.to_dict(),
                    "answer": answer,
                    "eval": judge_res
                }
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
