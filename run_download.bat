@echo off
REM Change to repository root (where this script is located)
cd /d %~dp0

REM Activate remote service venv
call services\remote\venv\Scripts\activate.bat

REM Run download script
python services\remote\download_sessions.py

