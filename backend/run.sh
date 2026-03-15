#!/bin/bash
# 1. Bật Backend (chạy ngầm)
python server.py &

# 2. Bật Frontend (Streamlit)
streamlit run main_web.py --server.port 8501 --server.address 0.0.0.0