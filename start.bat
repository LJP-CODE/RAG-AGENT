@echo off
chcp 65001 >nul
title AI Agent 问答 API

cd /d "%~dp0"

echo ============================================================
echo   AI Agent 问答 API 一键启动
echo ============================================================

REM ── 加载 .env 文件中的环境变量 ──
if exist ".env" (
    echo [1/4] 加载 .env 环境变量...
    for /f "usebackq tokens=1,2 delims== eol=#" %%a in (".env") do (
        set "%%a=%%b"
    )
) else (
    echo [警告] .env 文件不存在，将使用系统环境变量
)

REM ── 激活虚拟环境 ──
if exist ".venv\Scripts\activate.bat" (
    echo [2/4] 激活虚拟环境...
    call .venv\Scripts\activate.bat
) else (
    echo [2/4] 未找到 .venv，使用系统 Python
)

REM ── 安装依赖（如需要）──
echo [3/4] 检查依赖...
pip install -q sse-starlette>=2.1.0 2>nul

REM ── 启动服务 ──
echo [4/4] 启动服务...
echo.
echo   地址: http://localhost:8000
echo   文档: http://localhost:8000/docs
echo   健康: http://localhost:8000/health
echo.
echo   按 Ctrl+C 停止服务
echo ============================================================
echo.

uvicorn app.agent_api:app --reload --host 0.0.0.0 --port 8000

pause
