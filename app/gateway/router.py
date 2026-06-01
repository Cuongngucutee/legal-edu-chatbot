"""
LawEdu AI — API Gateway Router.
Handles all HTTP endpoints: chat, session, health, stats.
SSE streaming for real-time responses.
"""
import os
import json
import time
import uuid
import asyncio
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.gateway.models import ChatRequest, ChatResponse, SessionResponse, HealthResponse, StatsResponse
from app.rag.context_builder import build_context, postprocess_citations, fmt_source
from app.llm.prompts import GENERATION_PROMPT, GENERATION_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Load doc_titles for full name resolution
DOC_TITLES = {}
_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_doc_titles_path = os.path.join(_base_dir, "data", "doc_titles.json")
if os.path.exists(_doc_titles_path):
    with open(_doc_titles_path, "r", encoding="utf-8") as f:
        DOC_TITLES = json.load(f)

router = APIRouter(prefix="/api")

# These will be set during app initialization
_pipeline = None
_sessions = {}
_cache = None
_metrics = None


def init_router(pipeline, cache, metrics):
    """Initialize router with pipeline and cache references."""
    global _pipeline, _cache, _metrics
    _pipeline = pipeline
    _cache = cache
    _metrics = metrics


def save_session(sid, question, resolved, answer, sources, intent, latency_ms):
    history_dir = os.path.join(_base_dir, "data", "chat_history")
    os.makedirs(history_dir, exist_ok=True)
    history_file = os.path.join(history_dir, f"{sid}.json")
    
    data = {
        "session_id": sid,
        "title": "Cuộc trò chuyện mới",
        "created_at": time.time(),
        "updated_at": time.time(),
        "messages": []
    }
    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            pass
            
    # Update title if it's new
    if (not data.get("messages") or data.get("title") == "Cuộc trò chuyện mới") and question:
        data["title"] = question[:40] + ("..." if len(question) > 40 else "")
        
    data["updated_at"] = time.time()
    if "created_at" not in data:
        data["created_at"] = time.time()
        
    user_msg = {
        "role": "user",
        "content": question,
        "resolved": resolved,
        "timestamp": time.time()
    }
    assistant_msg = {
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "intent": intent,
        "latency_ms": latency_ms,
        "timestamp": time.time()
    }
    data["messages"].extend([user_msg, assistant_msg])
    
    try:
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving chat history for {sid}: {e}")


def load_session_history_if_exists(sid, mgr):
    history_file = os.path.join(_base_dir, "data", "chat_history", f"{sid}.json")
    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            messages = data.get("messages", [])
            mgr.history = []
            for i in range(0, len(messages) - 1, 2):
                if messages[i]["role"] == "user" and messages[i+1]["role"] == "assistant":
                    mgr.history.append({
                        "q": messages[i]["content"],
                        "resolved": messages[i].get("resolved", messages[i]["content"]),
                        "a": messages[i+1]["content"]
                    })
        except Exception as e:
            logger.warning(f"Error loading history for session {sid}: {e}")


@router.post("/session", response_model=SessionResponse)
async def new_session():
    from app.rag.pipeline import ConversationManager
    sid = str(uuid.uuid4())[:8]
    _sessions[sid] = ConversationManager(_pipeline)
    return SessionResponse(session_id=sid)


@router.get("/history")
async def list_history():
    history_dir = os.path.join(_base_dir, "data", "chat_history")
    if not os.path.exists(history_dir):
        return []
    
    sessions = []
    for filename in os.listdir(history_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(history_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sessions.append({
                    "session_id": data.get("session_id"),
                    "title": data.get("title", "Cuộc trò chuyện mới"),
                    "created_at": data.get("created_at", 0),
                    "updated_at": data.get("updated_at", 0)
                })
            except Exception as e:
                logger.warning(f"Error reading session file {filename}: {e}")
                
    sessions.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
    return sessions


@router.get("/history/{session_id}")
async def get_history_detail(session_id: str):
    history_file = os.path.join(_base_dir, "data", "chat_history", f"{session_id}.json")
    if not os.path.exists(history_file):
        if session_id in _sessions:
            return {
                "session_id": session_id,
                "title": "Cuộc trò chuyện mới",
                "created_at": time.time(),
                "updated_at": time.time(),
                "messages": []
            }
        raise HTTPException(status_code=404, detail="Session not found")
        
    try:
        with open(history_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading history: {e}")


@router.delete("/history/{session_id}")
async def delete_history(session_id: str):
    history_file = os.path.join(_base_dir, "data", "chat_history", f"{session_id}.json")
    if os.path.exists(history_file):
        try:
            os.remove(history_file)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting history file: {e}")
            
    if session_id in _sessions:
        del _sessions[session_id]
        
    return {"status": "success", "message": f"Session {session_id} deleted"}


@router.post("/chat")
async def chat(request: ChatRequest):
    if _pipeline is None:
        raise HTTPException(500, "Pipeline not initialized")

    from app.rag.pipeline import ConversationManager

    sid = request.session_id or "default"
    if sid not in _sessions:
        _sessions[sid] = ConversationManager(_pipeline)
        load_session_history_if_exists(sid, _sessions[sid])
    mgr = _sessions[sid]

    # Check cache
    cached = None
    if sid == "default" or not mgr.history:
        cached = _cache.get(request.question) if _cache else None

    if cached:
        print(f"👉 BẮT ĐẦU VÀO LUỒNG CACHE CHO CÂU HỎI: {request.question}", flush=True)
        if _metrics:
            _metrics.record_request(cached.get("intent", "LOOKUP"), 0.1, cache_hit=True)
        
        # Thêm vào history và lưu lịch sử
        mgr.history.append({"q": request.question, "resolved": request.question, "a": cached["answer"]})
        save_session(sid, request.question, request.question, cached["answer"], cached.get("sources", []), cached.get("intent"), 0.1)
        
        if request.stream:
            return StreamingResponse(
                _stream_cached(cached),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
            )
        else:
            resp = _build_response(cached, time.time(), request)
            resp.latency_ms = 0.1
            return resp

    if request.stream:
        return StreamingResponse(
            _stream_response(mgr, request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    start = time.time()
    result = mgr.chat(request.question, is_pro=request.is_pro)
    
    # Lưu lịch sử chat
    resolved = mgr.history[-1]["resolved"] if mgr.history else request.question
    save_session(sid, request.question, resolved, result["answer"], result.get("sources", []), result.get("intent"), (time.time() - start) * 1000)
    
    if sid == "default" or len(mgr.history) <= 1:
        if _cache:
            _cache.put(request.question, result)
    if _metrics:
        _metrics.record_request(result.get("intent", "LOOKUP"), (time.time() - start) * 1000)
    return _build_response(result, start, request)


def _build_response(result, start, request):
    latency = (time.time() - start) * 1000
    skill_used = None
    if result.get("skill_log"):
        skill_used = result["skill_log"][-1].get("skill")
    warning = None
    for src in result.get("sources", []):
        if src.get("tinh_trang") == "da_sua_doi":
            warning = "⚠️ Lưu ý: Một số văn bản được trích dẫn đã được sửa đổi, bổ sung."
            break
    return ChatResponse(
        answer=result["answer"],
        sources=result.get("sources", []) if request.include_sources else [],
        intent=result.get("intent"), skill_used=skill_used, warning=warning,
        iterations=result.get("iterations", 1), latency_ms=latency,
    )


async def _stream_cached(cached):
    """Stream a cached response with typing effect."""
    meta_data = json.dumps({"type": "meta", "intent": cached.get("intent", "LOOKUP")}, ensure_ascii=False)
    yield f"data: {meta_data}\n\n"

    sources = cached.get("sources", [])
    src_data = json.dumps({"type": "sources", "sources": sources}, ensure_ascii=False)
    yield f"data: {src_data}\n\n"

    words = cached["answer"].split(" ")
    for i, word in enumerate(words):
        tok = word + (" " if i < len(words) - 1 else "")
        tok_data = json.dumps({"type": "token", "content": tok}, ensure_ascii=False)
        yield f"data: {tok_data}\n\n"
        await asyncio.sleep(0.015)

    done_data = json.dumps({
        "type": "done", "answer": cached["answer"], "sources": sources,
        "intent": cached.get("intent"), "latency_ms": 0.1,
    }, ensure_ascii=False)
    yield f"data: {done_data}\n\n"


def _stream_response(mgr, request):
    """Stream a fresh response with real LLM token streaming."""
    from app.query.intent_classifier import classify_intent
    from app.query.query_expander import EducationQueryExpander

    pipeline = mgr.pipeline
    llm_client = pipeline.pro_generator_llm if (request.is_pro and hasattr(pipeline, "pro_generator_llm")) else pipeline.generator_llm
    start = time.time()

    # Resolve coreferences
    resolved = request.question
    
    from app.query.intent_classifier import get_fast_intent, classify_intent
    fast_intent = get_fast_intent(request.question)
    
    if mgr.history and not fast_intent:
        yield f"data: {json.dumps({'type': 'status', 'content': 'Đang phân tích ngữ cảnh...'}, ensure_ascii=False)}\n\n"
        from app.llm.prompts import CONVERSATION_RESOLVE_PROMPT
        hist = "\n".join(f"User: {h['q']}\nBot: {h['a'][:200]}..." for h in mgr.history[-3:])
        resolved = pipeline.llm.generate(
            CONVERSATION_RESOLVE_PROMPT.format(history=hist, new_query=request.question)
        ).strip()
        if len(resolved) < 3:
            resolved = request.question

    # Intent classification
    intent = classify_intent(resolved, pipeline.llm)
    meta_evt = json.dumps({"type": "meta", "intent": intent}, ensure_ascii=False)
    yield f"data: {meta_evt}\n\n"

    # Instant intents
    instant = {
        "GREETING": "Xin chào! 👋 Tôi là trợ lý AI chuyên về pháp luật giáo dục VN.\nHãy đặt câu hỏi để tôi giúp bạn!",
        "THANKS": "Không có gì! 😊 Nếu bạn có thêm câu hỏi, đừng ngần ngại hỏi nhé.",
        "UNANSWERABLE": "Câu hỏi này nằm ngoài phạm vi cơ sở dữ liệu pháp luật giáo dục VN.",
    }
    if intent in instant:
        text = instant[intent]
        words = text.split(" ")
        for i, word in enumerate(words):
            tok = word + (" " if i < len(words) - 1 else "")
            tok_data = json.dumps({"type": "token", "content": tok}, ensure_ascii=False)
            yield f"data: {tok_data}\n\n"
            time.sleep(0.02)
        mgr.history.append({"q": request.question, "resolved": resolved, "a": text})
        done_data = json.dumps({
            "type": "done", "sources": [], "latency_ms": (time.time() - start) * 1000,
            "intent": intent, "answer": text,
        }, ensure_ascii=False)
        yield f"data: {done_data}\n\n"
        return

    # ── STAGE 0: Wide Retrieval + RRF Fusion ────────────────────
    yield f"data: {json.dumps({'type': 'status', 'content': 'Đang tra cứu dữ liệu pháp luật ban đầu...'}, ensure_ascii=False)}\n\n"
    import re
    import os
    import math
    import datetime
    os.makedirs("outputs", exist_ok=True)
    trace_path = "outputs/agentic_rag_trace.txt"
    now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Helper: extract bare so_hieu from ten_van_ban
    def _bare_sh(text):
        """'Thông tư 01/2021/TT-BGDĐT' → '01/2021/TT-BGDĐT'"""
        m = re.search(r'(\d+[\/\-]\d{4}[\/\-]?[\w\-]*)', str(text))
        return m.group(1) if m else text

    def _resolve_sh(raw_sh, registry):
        """Resolve so_hieu to doc_registry key. Exact match first, then safe fuzzy."""
        if raw_sh in registry:
            return raw_sh
        for variant in [raw_sh.lower(), raw_sh.replace('-', '/'), raw_sh.replace('/', '-')]:
            if variant in registry:
                return variant
        for k in registry:
            if len(k) > 3 and len(raw_sh) > 3:
                if k == raw_sh or (raw_sh.startswith(k.split('/')[0]) and raw_sh.endswith(k.split('/')[-1]) and abs(len(k)-len(raw_sh)) < 5):
                    return k
        return raw_sh

    def _rrf_merge(ranked_lists, k=60):
        """Reciprocal Rank Fusion: merge multiple ranked lists into one."""
        scores = {}  # chunk_id → cumulative RRF score
        chunk_map = {}  # chunk_id → doc dict
        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked):
                cid = doc.get("chunk_id", "")
                if not cid:
                    continue
                scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
                if cid not in chunk_map:
                    chunk_map[cid] = doc
        # Sort by RRF score descending
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [chunk_map[cid] for cid in sorted_ids], scores

    # Wide retrieval: main query + expanded queries, merged via RRF
    main_docs = pipeline.retriever.retrieve_as_docs(resolved, top_k=20)
    all_ranked_lists = [main_docs]

    expander = EducationQueryExpander()
    expanded = expander.expand(resolved)
    
    import concurrent.futures
    def fetch_eq(eq):
        return pipeline.retriever.retrieve_as_docs(eq, top_k=8)
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(fetch_eq, expanded[:3]))
        for eq_docs in results:
            if eq_docs:
                all_ranked_lists.append(eq_docs)

    docs, rrf_scores = _rrf_merge(all_ranked_lists)

    if not docs:
        answer = "Không tìm thấy quy định pháp luật giáo dục cụ thể."
        yield f"data: {json.dumps({'type': 'token', 'content': answer}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'sources': [], 'latency_ms': (time.time()-start)*1000, 'intent': intent}, ensure_ascii=False)}\n\n"
        return

    # ── STAGE 1: Doc-Level Vote Aggregation + 7B Selection ───
    yield f"data: {json.dumps({'type': 'status', 'content': 'Đang phân tích và chọn lọc văn bản pháp luật...'}, ensure_ascii=False)}\n\n"

    # Aggregate at document level: count chunks + sum RRF scores per VB
    doc_agg = {}  # bare_so_hieu → {ten, count, total_rrf, max_rrf}
    for d in docs:
        meta = d.get("metadata", {})
        ten = meta.get("ten_van_ban") or meta.get("source") or ""
        bare = _bare_sh(ten)
        cid = d.get("chunk_id", "")
        rrf = rrf_scores.get(cid, 0)
        if bare:
            if bare not in doc_agg:
                doc_agg[bare] = {"ten": ten, "count": 0, "total_rrf": 0, "max_rrf": 0}
            doc_agg[bare]["count"] += 1
            doc_agg[bare]["total_rrf"] += rrf
            doc_agg[bare]["max_rrf"] = max(doc_agg[bare]["max_rrf"], rrf)

    # Rank documents: combined score = max_rrf + 0.3 * log(1 + count) * avg_rrf
    # This boosts documents with multiple relevant chunks
    unique_docs = {}
    if doc_agg:
        for bare, info in doc_agg.items():
            avg_rrf = info["total_rrf"] / info["count"] if info["count"] else 0
            info["rank_score"] = info["max_rrf"] + 0.3 * math.log(1 + info["count"]) * avg_rrf
            unique_docs[bare] = info["ten"]

        ranked_docs = sorted(doc_agg.items(), key=lambda x: x[1]["rank_score"], reverse=True)

    top_so_hieu = []
    if unique_docs:
        # Build context for 7B: show top-ranked docs with vote info
        top_candidates = ranked_docs[:8]  # show top 8 to 7B
        doc_options = "\n".join([
            f"- {sh} ({info['ten']}) [chunks={info['count']}, score={info['rank_score']:.3f}]"
            for sh, info in top_candidates
        ])
        # Use top-20 docs for context snippets (already RRF-ranked)
        context_text = "\n".join([
            f"[{d.get('metadata',{}).get('ten_van_ban','')}] Điều {d.get('metadata',{}).get('so_dieu','')}: {(d.get('content','') or d.get('text',''))[:150]}..."
            for d in docs[:20]
        ])
        filter_prompt = f"""Dựa vào câu hỏi: "{resolved}"
Các trích đoạn (đã xếp hạng theo độ liên quan):
{context_text}

Danh sách văn bản (xếp hạng theo mức liên quan):
{doc_options}

Chọn TỐI ĐA 2 số hiệu văn bản liên quan nhất để trả lời câu hỏi.
Chỉ xuất mảng JSON. Ví dụ: ["02/2021/TT-BGDĐT", "13/2024/TT-BGDĐT"]"""

        try:
            import ast
            filter_res = llm_client.generate(filter_prompt, temperature=0.1)
            match = re.search(r'\[(.*?)\]', filter_res)
            if match:
                parsed = ast.literal_eval(f"[{match.group(1)}]")
                for p in parsed:
                    bare_p = _bare_sh(p)
                    if bare_p in unique_docs:
                        top_so_hieu.append(bare_p)
                    else:
                        for k in unique_docs:
                            if bare_p == k or (len(bare_p) > 5 and len(k) > 5 and (bare_p in k or k in bare_p)):
                                top_so_hieu.append(k)
                                break
            if not top_so_hieu:
                # Fallback: use top-2 by rank_score
                top_so_hieu = [sh for sh, _ in ranked_docs[:2]]
        except Exception:
            top_so_hieu = [sh for sh, _ in ranked_docs[:2]]

    # Write Stage 1 trace
    if top_so_hieu:
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(f"\n[{now_str}] Câu hỏi: {request.question}\n")
            f.write(f"👉 [Stage 1 - 7B] Văn bản được chọn:\n")
            for sh in top_so_hieu:
                f.write(f"   - {unique_docs.get(sh, sh)} ({sh})\n")

    # ── STAGE 2: 320B phân tích mục lục (with retry) ────────
    yield f"data: {json.dumps({'type': 'status', 'content': 'Đang soi chiếu Mục lục...'}, ensure_ascii=False)}\n\n"

    toc_parts = []
    for sh in top_so_hieu:
        toc = pipeline.retriever.index.build_toc(sh, query=resolved)
        if toc:
            toc_parts.append(toc)

    final_node_ids = set()
    stage2_selections = []

    def _parse_pro_articles(response_text, toc_so_hieus):
        """Parse 320B response into (node_ids, selections)."""
        node_ids = set()
        selections = []
        m = re.search(r'\[.*\]', response_text, re.DOTALL)
        if not m:
            return node_ids, selections
        try:
            items = json.loads(m.group(0))
        except json.JSONDecodeError:
            return node_ids, selections
        for item in items:
            raw_sh = _bare_sh(item.get("so_hieu", ""))
            dieu = item.get("dieu")
            if not raw_sh or dieu is None:
                continue
            resolved_sh = _resolve_sh(raw_sh, pipeline.retriever.index.doc_registry)
            doc_node_ids = pipeline.retriever.index.get_doc_node_ids(resolved_sh)
            for nid in doc_node_ids:
                if nid not in pipeline.retriever.index.graph.nodes:
                    continue
                nd = pipeline.retriever.index.graph.nodes[nid]
                name = nd.get("name", "")
                if not name:
                    ft = nd.get("full_text", "") or nd.get("search_text", "")
                    if ft:
                        name = ft.split("\n")[0].strip()
                dm = re.search(r'Điều\s+(\d+)', name)
                if dm and int(dm.group(1)) == int(dieu):
                    node_ids.add(nid)
                    selections.append(f"{unique_docs.get(raw_sh, raw_sh)} - Điều {dieu}")
        return node_ids, selections

    if toc_parts:
        toc_text = "\n\n".join(toc_parts)

        # ── Attempt 1: Standard prompt ──
        pro_prompt = f"""Câu hỏi: "{resolved}"

Mục lục các văn bản:
{toc_text}

Nhiệm vụ: Chọn các Điều khoản chứa thông tin để trả lời câu hỏi.
Trả về JSON: [{{"so_hieu": "...", "dieu": <số>}}]
Chỉ xuất mảng JSON, không giải thích."""

        try:
            pro_res = pipeline.llm.generate(pro_prompt, temperature=0.1)
            final_node_ids, stage2_selections = _parse_pro_articles(pro_res, top_so_hieu)
        except Exception as e:
            with open(trace_path, "a", encoding="utf-8") as f:
                f.write(f"⚠️ [Stage 2 Attempt 1 Error] {e}\n")

        # ── Attempt 2: Retry with softer prompt if <2 articles ──
        if len(final_node_ids) < 2:
            retry_prompt = f"""Câu hỏi: "{resolved}"

Mục lục các văn bản:
{toc_text}

Nhiệm vụ: Liệt kê TẤT CẢ các Điều khoản có thể liên quan đến câu hỏi, kể cả các điều liên quan gián tiếp (ví dụ: điều định nghĩa, điều quy định phạm vi áp dụng, điều quy định đối tượng).
Chọn ÍT NHẤT 2 điều khoản.
Trả về JSON: [{{"so_hieu": "...", "dieu": <số>}}]
Chỉ xuất mảng JSON, không giải thích."""

            try:
                res_retry = pipeline.llm.generate(retry_prompt, temperature=0.2)
                retry_nodes, retry_selections = _parse_pro_articles(res_retry, top_so_hieu)
                if len(retry_nodes) > len(final_node_ids):
                    final_node_ids = retry_nodes
                    stage2_selections = retry_selections
            except Exception:
                pass

        # ── Attempt 3: Expand to next-ranked docs if still empty ──
        if not final_node_ids and len(ranked_docs) > 2:
            extra_so_hieus = [sh for sh, _ in ranked_docs[2:5] if sh not in top_so_hieu]
            extra_tocs = []
            for sh in extra_so_hieus:
                toc = pipeline.retriever.index.build_toc(sh)
                if toc:
                    extra_tocs.append(toc)
            if extra_tocs:
                expanded_toc = "\n\n".join(toc_parts + extra_tocs)
                expand_prompt = f"""Câu hỏi: "{resolved}"

Mục lục các văn bản:
{expanded_toc}

Nhiệm vụ: Chọn các Điều khoản chứa thông tin để trả lời câu hỏi.
Chọn ÍT NHẤT 2 điều khoản.
Trả về JSON: [{{"so_hieu": "...", "dieu": <số>}}]
Chỉ xuất mảng JSON, không giải thích."""

                try:
                    res_expand = pipeline.llm.generate(expand_prompt, temperature=0.2)
                    expand_nodes, expand_selections = _parse_pro_articles(res_expand, top_so_hieu + extra_so_hieus)
                    if expand_nodes:
                        final_node_ids = expand_nodes
                        stage2_selections = expand_selections
                except Exception:
                    pass

    # Write Stage 2 trace
    with open(trace_path, "a", encoding="utf-8") as f:
        if stage2_selections:
            f.write(f"👉 [Stage 2 - 320B] Các Điều khoản được chọn lọc:\\n")
            for sel in stage2_selections:
                f.write(f"   - {sel}\n")
        else:
            f.write(f"⚠️ [Stage 2 - 320B] Không tìm thấy điều khoản phù hợp, dùng retrieval gốc\n")
        f.write(f"{'-'*60}\n")

    # ── STAGE 3: Build final context ───────────────────────
    if final_node_ids:
        yield f"data: {json.dumps({'type': 'status', 'content': 'Đang trích xuất chi tiết các Điều khoản...'}, ensure_ascii=False)}\n\n"
        final_docs = []
        for nid in final_node_ids:
            d = pipeline.retriever._node_to_doc(nid)
            if d:
                final_docs.append(d)
        if final_docs:
            docs = final_docs
            
    sources = [fmt_source(d) for d in docs[:8]]
    yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"

    # Stream LLM generation
    yield f"data: {json.dumps({'type': 'status', 'content': '⏳ Đang tổng hợp câu trả lời...'}, ensure_ascii=False)}\n\n"
    context = build_context(docs[:10])
    prompt = GENERATION_PROMPT.format(query=resolved, context=context)

    full_answer = []
    for token in llm_client.generate_stream(prompt, system_prompt=GENERATION_SYSTEM_PROMPT):
        full_answer.append(token)
        yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"

    answer = "".join(full_answer).strip()
    answer = postprocess_citations(answer, docs)
    
    # ── Chèn thêm chú thích (Footnote) bằng code Python ──
    unique_so_hieu = []
    for s in sources:
        # Lấy bất kỳ thông tin nào có thể nhận diện được văn bản
        sh = s.get("so_hieu") or s.get("doc_id") or s.get("ten_van_ban")
        if sh and sh not in unique_so_hieu:
            unique_so_hieu.append(sh)
            
    with open("debug_footnote.txt", "w", encoding="utf-8") as f:
        f.write(f"unique_so_hieu: {unique_so_hieu}\n")
        f.write(f"DOC_TITLES size: {len(DOC_TITLES)}\n")
            
    if unique_so_hieu and DOC_TITLES:
        footnote_lines = ["\n\n---\n**Chú thích Tên văn bản đầy đủ:**\n"]
        has_note = False
        seen_notes = set()
        for sh in unique_so_hieu:
            title = DOC_TITLES.get(sh)
            display_sh = sh
            
            # Cố gắng trích xuất mẫu Số/Năm (vd: 238_2025 -> 238/2025)
            import re
            if not title:
                m = re.search(r'(\d+)[_/-](\d{4})', sh)
                if m:
                    pattern = f"{m.group(1)}/{m.group(2)}"
                    for k, v in DOC_TITLES.items():
                        if pattern in k:
                            title = v
                            display_sh = k
                            break
                            
            # Nếu là Luật (vd: Luật43, Luat 43)
            if not title:
                m2 = re.search(r'Lu[aậ]t\s*(\d+)', sh, re.IGNORECASE)
                if m2:
                    pattern = f"Luật {m2.group(1)}"
                    for k, v in DOC_TITLES.items():
                        if k.startswith(pattern):
                            title = v
                            display_sh = k
                            break

            # Thử tìm kiếm gần đúng (substring match) nếu vẫn chưa thấy
            if not title:
                for k, v in DOC_TITLES.items():
                    if (k in sh or sh in k) and len(k) > 5:
                        title = v
                        display_sh = k
                        break
            
            with open("debug_footnote.txt", "a", encoding="utf-8") as f:
                f.write(f"sh: {sh} -> title: {title}\n")

            if title and title not in seen_notes:
                footnote_lines.append(f"- **{display_sh}**: {title}\n")
                seen_notes.add(title)
                has_note = True
                
        if has_note:
            footnote_text = "".join(footnote_lines)
            answer += footnote_text
            yield f"data: {json.dumps({'type': 'token', 'content': footnote_text}, ensure_ascii=False)}\n\n"
    # ──────────────────────────────────────────────────────

    mgr.history.append({"q": request.question, "resolved": resolved, "a": answer})
    save_session(request.session_id or "default", request.question, resolved, answer, sources, intent, (time.time() - start) * 1000)

    # Cache the result
    result = {"answer": answer, "sources": sources, "intent": intent}
    if _cache and (request.session_id is None or request.session_id == "default"):
        _cache.put(request.question, result)
    if _metrics:
        _metrics.record_request(intent, (time.time() - start) * 1000)

    warning = None
    for s in sources:
        if s.get("tinh_trang") == "da_sua_doi":
            warning = "⚠️ Lưu ý: Một số văn bản được trích dẫn đã được sửa đổi, bổ sung."
            break

    yield f"data: {json.dumps({'type': 'done', 'answer': answer, 'sources': sources, 'intent': intent, 'warning': warning, 'is_pro': request.is_pro, 'latency_ms': (time.time()-start)*1000}, ensure_ascii=False)}\n\n"


@router.get("/health")
async def health():
    return HealthResponse(
        status="ok",
        pipeline=_pipeline is not None,
        llm_model=_pipeline.llm.model if _pipeline else None,
        cache_backend=_cache.stats().get("backend") if _cache else None,
    )


@router.get("/stats")
async def stats():
    if _pipeline is None:
        return {"error": "Not initialized"}
    n_nodes = _pipeline.retriever.index.graph.number_of_nodes() if hasattr(_pipeline.retriever, 'index') else 0
    return StatsResponse(
        sessions=len(_sessions),
        nodes=n_nodes,
        model=_pipeline.llm.model,
        cache=_cache.stats() if _cache else {},
        metrics=_metrics.to_dict() if _metrics else {},
    )
