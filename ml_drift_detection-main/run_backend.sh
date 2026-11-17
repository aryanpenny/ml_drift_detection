#!/bin/bash
# ==============================================================================
# DriftGuard — FastAPI Backend Model Serving Launcher
# ==============================================================================

set -e

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_DIR"

echo "=================================================================="
echo "  🚀 Launching FastAPI Prediction Serving Backend"
echo "=================================================================="

if [ -f "/opt/homebrew/opt/python@3.11/bin/python3.11" ]; then
    PYTHON_EXEC="/opt/homebrew/opt/python@3.11/bin/python3.11"
    UVICORN_EXEC="/opt/homebrew/bin/uvicorn"
else
    PYTHON_EXEC="$(command -v python3)"
    UVICORN_EXEC="$(command -v uvicorn)"
fi

export PYTHONPATH="$PROJECT_DIR"
echo "Starting FastAPI on http://localhost:8000 (Swagger docs: http://localhost:8000/docs)..."
$UVICORN_EXEC src.serve:app --host 0.0.0.0 --port 8000 --reload
