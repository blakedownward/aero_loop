@echo off
REM Change to repository root (where this script is located)
cd /d %~dp0

REM Activate annotator service venv
call services\annotator\venv\Scripts\activate.bat

REM Run annotator app
streamlit run services\annotator\app.py

