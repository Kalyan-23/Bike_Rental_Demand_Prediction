@echo off
echo Starting Seoul Bike Sharing Demand Prediction System...
cd /d "%~dp0"
python -m streamlit run app.py
pause

