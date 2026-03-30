import json
from pathlib import Path

import streamlit as st

from test_graphrag_kg_full import KG_PATH, KGFullGraphRAGTester

st.set_page_config(page_title="Legal GraphRAG Explorer", layout="wide")

st.title("Legal GraphRAG Explorer")
st.caption("Quan sat truy xuat hybrid: title + graph relations + clause content")

if not KG_PATH.exists():
    st.error(f"Khong tim thay KG: {KG_PATH}")
    st.stop()

@st.cache_resource
def load_tester(kg_file: str):
    return KGFullGraphRAGTester(Path(kg_file))


tester = load_tester(str(KG_PATH))

with st.sidebar:
    st.header("Cau hinh truy van")
    top_k = st.slider("Top-K", min_value=1, max_value=20, value=5, step=1)
    show_raw = st.checkbox("Hien thi JSON thuan", value=False)

question = st.text_area(
    "Nhap cau hoi",
    value="Nhung dieu nao lien quan den sua doi, bo sung va bai bo trong Luat Giao duc dai hoc?",
    height=90,
)

run = st.button("Chay truy van", type="primary")

if run and question.strip():
    result = tester.ask(question.strip(), top_k=top_k)
    rows = result.get("top_articles", [])

    st.subheader("Ket qua")
    if not rows:
        st.warning("Khong tim thay ung vien phu hop.")
    else:
        summary = [
            {
                "rank": idx,
                "article_number": item.get("article_number", ""),
                "title": item.get("title", ""),
                "base_score": item.get("base_score", 0.0),
                "final_score": item.get("score", 0.0),
            }
            for idx, item in enumerate(rows, start=1)
        ]
        st.dataframe(summary, use_container_width=True, hide_index=True)

        for idx, item in enumerate(rows, start=1):
            with st.expander(f"#{idx} - Dieu {item.get('article_number', '?')}: {item.get('title', '')}"):
                st.write(f"Base score: {item.get('base_score', 0.0)}")
                st.write(f"Final score (after rerank): {item.get('score', 0.0)}")
                reasons = item.get("reasons", [])
                if reasons:
                    st.markdown("**Ly do cham diem**")
                    for r in reasons:
                        st.write(f"- {r}")

                clauses = item.get("sample_clauses", [])
                if clauses:
                    st.markdown("**Clause content retrieval (BM25 + semantic)**")
                    for c in clauses:
                        st.write(
                            f"- {c.get('clause_id', '')} | bm25={c.get('bm25', 0)} | semantic={c.get('semantic', 0)} | hybrid={c.get('hybrid', 0)}"
                        )
                        st.code(c.get("text_preview", ""))

                rels = item.get("sample_relations", [])
                if rels:
                    st.markdown("**Graph relations**")
                    for rel in rels:
                        st.write(
                            f"- {rel.get('subject', '')} --{rel.get('relation', '')}--> {rel.get('object', '')}"
                        )

                evs = item.get("sample_evidence", [])
                if evs:
                    st.markdown("**Evidence**")
                    for ev in evs:
                        st.write(f"- [{ev.get('relation', '')}] {ev.get('quote', '')}")

        if show_raw:
            st.subheader("Raw JSON")
            st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")
else:
    st.info("Nhap cau hoi va bam 'Chay truy van' de xem ket qua.")
