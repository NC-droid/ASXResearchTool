@echo off
REM Double-click this file to launch the ASX ETF Screener on Windows.

cd /d "%~dp0"

if not exist ".venv\" (
  echo [ASX ETF Screener] Creating virtualenv...
  python -m venv .venv
)

call .venv\Scripts\activate.bat

if not exist ".venv\.install-stamp" (
  echo [ASX ETF Screener] Installing dependencies...
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  echo. > .venv\.install-stamp
)

echo [ASX ETF Screener] Starting server at http://127.0.0.1:8000/
start "" http://127.0.0.1:8000/
uvicorn asx_portfolio.web_app:app --host 127.0.0.1 --port 8000
