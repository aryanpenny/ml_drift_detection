# DriftGuard: Closed-Loop Drift Detection & Automated Retraining MLOps Platform

**DriftGuard** is a closed-loop Machine Learning Operations (MLOps) platform designed to detect statistical data drift in production streams, benchmark multiple drift algorithms, and automate model retraining and redeployment.

---

## 🌟 Key Features

1. **Multi-Algorithm Drift Detection**:
   - **Kolmogorov-Smirnov (KS) Test**: Non-parametric two-sample test for exact p-value significance.
   - **Population Stability Index (PSI)**: Standard binned divergence score with industry severity scales (No Drift, Moderate, High).
   - **Adaptive Windowing (ADWIN)**: Online streaming drift detection using River, tracking sequential variance shifts and detection latency.
2. **Benchmark Suite**:
   - Empirical evaluation across controlled scenarios: *Stable Baseline*, *Sudden Shift*, *Gradual Linear Drift*, and *Variance Expansion*.
   - Measurement of execution time, latency, false-positive rate, and ground-truth accuracy.
3. **Automated Closed-Loop Retraining**:
   - Automated retraining condition validation (`should_retrain`).
   - Integration with Apache Airflow `BranchPythonOperator` and MLflow tracking.
4. **FastAPI Model Serving**:
   - High-throughput inference server exposing `/` (health check) and `/predict` endpoints.
5. **Premium Streamlit UI**:
   - Linear/Vercel-inspired dark/light theme switching.
   - 8 comprehensive pages: Overview, Drift Monitoring, Detection Benchmark, Model Performance, Retraining Pipeline, Model Management, Data Management, and Settings.

---

## 🚀 How to Run Front-End and Back-End

### 1. Launch the Frontend (Streamlit Dashboard)
```bash
./run_app.sh
```
*Alternatively, run directly:*
```bash
PYTHONPATH=. streamlit run src/dashboard.py
```
Open your browser at: **`http://localhost:8501`**

### 2. Launch the Backend (FastAPI Model Serving)
```bash
./run_backend.sh
```
*Alternatively, run directly:*
```bash
PYTHONPATH=. uvicorn src.serve:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API docs (Swagger): **`http://localhost:8000/docs`**

---

## 🧪 Testing
Run the complete test suite:
```bash
pytest tests/
```

---

## 📁 Project Structure
```
├── dags/
│   └── churn_pipeline.py     # Airflow retraining DAG
├── data/
│   ├── raw/                  # Raw Telco customer churn dataset
│   ├── chunks/               # Chunked stream partitions (Chunk 1=Ref, 2..4=Stream)
│   └── processed/            # Preprocessed train/test matrices
├── models/
│   └── churn_model.pkl       # Trained Random Forest model
├── src/
│   ├── ingest.py             # Data ingestion and chunk partitioning
│   ├── preprocess.py         # LabelEncoding and train/test splitting
│   ├── train.py              # MLflow-tracked model training & evaluation
│   ├── drift.py              # KS, PSI, and River ADWIN drift detectors
│   ├── benchmark.py          # Benchmark suite with controlled drift scenarios
│   ├── monitoring.py         # State manager for pipeline runs and drift events
│   ├── serve.py              # FastAPI inference endpoints
│   └── dashboard.py          # DriftGuard Streamlit SaaS interface
├── tests/
│   └── test_drift_algorithms.py # Pytest test suite for drift algorithms
├── run_app.sh                # Frontend runner script
├── run_backend.sh            # Backend runner script
├── requirements.txt          # Dependency specification
└── README.md
```
