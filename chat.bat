@echo off
chcp 65001 >nul
cd /d "%~dp0"
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" chat_cli.py
) else (
    python chat_cli.py
)
pause
