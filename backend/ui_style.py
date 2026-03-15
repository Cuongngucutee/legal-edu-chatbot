import streamlit as st

def apply_custom_styles():
    st.markdown("""
        <style>
        .main { background-color: #f8f9fa; }
        .stChatMessage { border-radius: 12px; border: 1px solid #ddd; }
        .stSidebar { background-color: #1e293b; color: white; }
        .stCaption { font-size: 0.8rem; color: #666; }
        </style>
    """, unsafe_allow_html=True)

def sidebar_content():
    with st.sidebar:
        st.title("⚖️ Pháp Luật AI")
        st.write("Phiên bản 2.0 (RAG + KG)")
        if st.button("Reset Chat"):
            st.session_state.messages = []
            st.rerun()