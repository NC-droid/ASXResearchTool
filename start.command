#!/usr/bin/env bash
# Double-click this file in Finder to launch the ASX ETF Screener.
#
# What it does:
#   1) cd into the project directory (the folder this file lives in)
#   2) ensure a Python virtualenv exists at .venv (creates on first run)
#   3) install/update dependencies if requirements.txt is newer than the venv
#   4) start uvicorn serving the FastAPI app on http://127.0.0.1:8000
#   5) WAIT until the server actually answers before opening the browser
#   6) press Ctrl-C in the Terminal window to stop the server.

set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"
REQ_FILE="requirements.txt"
INSTALL_STAMP=".venv/.install-stamp"
URL="http://127.0.0.1:8000/"

echo "[ASX ETF Screener] Project: $DIR"

# --- Sanity check: Python 3 must be available -------------------------------
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo
  echo "ERROR: '$PYTHON_BIN' not found on PATH."
  echo "Install Python 3 from https://www.python.org/downloads/ and try again."
  echo
  read -p "Press Enter to close this window..." _
  exit 1
fi

# --- Virtualenv -------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
  echo "[ASX ETF Screener] Creating virtualenv (one-time)..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# --- Dependencies -----------------------------------------------------------
if [ ! -f "$INSTALL_STAMP" ] || [ "$REQ_FILE" -nt "$INSTALL_STAMP" ]; then
  echo "[ASX ETF Screener] Installing dependencies — this may take 1-2 minutes the first time..."
  pip install --upgrade pip >/dev/null
  if ! pip install -r "$REQ_FILE"; then
    echo
    echo "ERROR: pip install failed. See the messages above."
    echo "Common causes: no internet connection, Python version <3.10, or a corrupt venv."
    read -p "Press Enter to close this window..." _
    exit 1
  fi
  touch "$INSTALL_STAMP"
fi

# --- Start uvicorn in the background, then wait for it to answer ------------
echo "[ASX ETF Screener] Starting server at $URL"
uvicorn asx_portfolio.web_app:app --host 127.0.0.1 --port 8000 &
SERVER_PID=$!

# Make sure we kill the server when the user closes the Terminal window
cleanup() {
  echo
  echo "[ASX ETF Screener] Stopping server (pid=$SERVER_PID)..."
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wait up to 60s for the server to start answering
echo "[ASX ETF Screener] Waiting for server to come up..."
for i in $(seq 1 120); do
  if curl -s -o /dev/null -w "%{http_code}" "$URL" 2>/dev/null | grep -qE "^(200|3[0-9]{2})$"; then
    echo "[ASX ETF Screener] Server is up after ${i} attempts"
    open "$URL" 2>/dev/null || true
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo
    echo "ERROR: uvicorn exited before answering. See the messages above."
    read -p "Press Enter to close this window..." _
    exit 1
  fi
  sleep 0.5
done

echo "[ASX ETF Screener] Press Ctrl-C to stop the server."
wait "$SERVER_PID"
