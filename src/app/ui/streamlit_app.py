from __future__ import annotations

import json
from typing import Dict

import streamlit as st

from app.core.config import AppConfig
from app.rag.service import RagService


st.set_page_config(page_title="Legal Edu Chatbot - RAG Debug", page_icon="⚖️", layout="wide")


@st.cache_resource
def get_rag_service() -> RagService:
    config = AppConfig.from_env()
    return RagService(config)


def render_hit_card(hit: Dict[str, str]) -> None:
    st.markdown(
        f"""
### Rank {hit['rank']} | score={hit['score']}
- Source: {hit['source']}
- Article: {hit['article_number']} - {hit['article_title']}
- Chapter/Section: {hit['chapter']} / {hit['section']}
"""
    )
    st.code(hit["text"], language="markdown")


def main() -> None:
    st.title("Legal Education Chatbot - RAG Monitor")
    st.caption("Retrieval-only mode: FAISS tim cac dieu khoan phu hop voi cau hoi")

    with st.sidebar:
        st.subheader("Cai dat")
        show_raw_trace = st.checkbox("Show raw trace JSON", value=False)

    question = st.text_area(
        "Nhap cau hoi",
        placeholder="Vi du: Dieu kien de mo nganh dao tao trinh do dai hoc la gi?",
        height=120,
    )

    if st.button("Run RAG", type="primary"):
        if not question.strip():
            st.warning("Vui long nhap cau hoi")
            return

        try:
            with st.spinner("Running retrieval..."):
                rag = get_rag_service()
                response = rag.ask(question.strip())
        except Exception as exc:
            st.error(f"Retrieval failed: {exc}")
            st.stop()

        st.success(f"Completed in {response.elapsed_ms} ms")

        st.subheader("Retrieved Contexts")
        hit_dicts = [h.to_debug_dict() for h in response.hits]
        for hit in hit_dicts:
            render_hit_card(hit)

        if show_raw_trace:
            st.subheader("Raw Retrieval Trace")
            st.json(
                {
                    "question": question.strip(),
                    "elapsed_ms": response.elapsed_ms,
                    "hits": hit_dicts,
                }
            )

        st.download_button(
            "Download latest debug trace",
            data=json.dumps(
                {
                    "question": question.strip(),
                    "elapsed_ms": response.elapsed_ms,
                    "hits": hit_dicts,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file_name="rag_debug_trace.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
