#!/bin/bash
# ==============================================================================
# DriftGuard — Local Application Launcher
# Closed-Loop Drift Detection and Automated Retraining Pipeline for ML Models
# ==============================================================================

set -e

# Resolve project directory
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_DIR"

echo "=================================================================="
echo "  🛡️  Launching DriftGuard MLOps Monitoring Application"
echo "=================================================================="
echo "Project Path: $PROJECT_DIR"

# 1. Detect Python 3.11 or active python
if [ -f "/opt/homebrew/opt/python@3.11/bin/python3.11" ]; then
    PYTHON_EXEC="/opt/homebrew/opt/python@3.11/bin/python3.11"
    STREAMLIT_EXEC="/opt/homebrew/bin/streamlit"
elif command -v python3 &>/dev/null; then
    PYTHON_EXEC="$(command -v python3)"
    STREAMLIT_EXEC="$(command -v streamlit)"
else
    echo "❌ Error: Python 3 not found!"
    exit 1
fi

echo "Using Python: $PYTHON_EXEC"

# 2. Check essential packages
echo "Checking required dependencies..."
$PYTHON_EXEC -c "
import streamlit
import pandas
import numpy
import scipy
import sklearn
import river
import plotly
import mlflow
import fastapi
print('✅ All core packages verified successfully.')
"

# 3. Ensure data chunks exist (bootstrap if missing)
if [ ! -f "data/chunks/chunk_1.csv" ]; then
    echo "Partitioning raw dataset into chunks..."
    PYTHONPATH=. $PYTHON_EXEC src/ingest.py
fi

if [ ! -f "models/churn_model.pkl" ]; then
    echo "Bootstrapping baseline model..."
    PYTHONPATH=. $PYTHON_EXEC src/preprocess.py 1
    PYTHONPATH=. $PYTHON_EXEC src/train.py
fi

# 4. Launch Streamlit Application
echo "Starting DriftGuard Streamlit Dashboard on http://localhost:8501 ..."
export PYTHONPATH="$PROJECT_DIR"
$STREAMLIT_EXEC run src/dashboard.py --server.port=8501 --server.headless=false
