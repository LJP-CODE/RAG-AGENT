@echo off
chcp 65001 >nul
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app\desktop_client.py
) else (
    python app\desktop_client.py
)
pause
