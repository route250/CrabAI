@echo off

@rem .venv\scripts\activate.bat

streamlit run ./src/crabai-st.py --server.address 0.0.0.0 --server.headless true 
pause
