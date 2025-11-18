@echo off
REM Change to repository root (where this script is located)
cd /d %~dp0

REM Activate mlops service venv
call services\mlops\venv\Scripts\activate.bat

REM Run orchestrator with train & deploy steps
python services\mlops\orchestrator.py --steps process_annotations upload_to_ei train_model evaluate_model build_and_download deploy_model --wait-training

