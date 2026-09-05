# Closed-Loop Drift Detection and Automated Retraining Pipeline

This implementation plan outlines the steps required to transform the current `ml_flow` project into the **Closed-Loop Drift Detection and Automated Retraining Pipeline** described in your document.

---

## Executive Summary & Feasibility

**Can you make this project like the abstract?**  
**YES, 100%.** The existing `ml_flow` codebase already provides ~60% of the required foundation:
- **MLflow** for experiment tracking and model registry
- **Airflow** for DAG execution
- **FastAPI / KServe** for serving predictions
- **DVC / Data Chunks** representing sequential time periods

To fulfill the abstract completely, we need to introduce a **Multi-Method Drift Engine**, a **Closed-Loop Trigger Mechanism**, and a **Streamlit Visualization Dashboard**.

---

## Key Differences & Required Changes

```
┌─────────────────────────────────────────┐      ┌─────────────────────────────────────────┐
│              Current ml_flow            │      │        Target Closed-Loop Pipeline      │
├─────────────────────────────────────────┼──────┼─────────────────────────────────────────┤
│ Retraining triggered manually / fixed   │ ───► │ Retraining triggered ONLY when drift    │
│ Airflow schedule                        │      │ crosses defined statistical threshold   │
├─────────────────────────────────────────┼──────┼─────────────────────────────────────────┤
│ Basic Prometheus metric monitoring      │ ───► │ Multi-method statistical & streaming    │
│ (latency & request counter)             │      │ drift detection (KS, PSI, ADWIN)        │
├─────────────────────────────────────────┼──────┼─────────────────────────────────────────┤
│ Grafana metrics view                    │ ───► │ Streamlit Dashboard showing drift, cost │
│                                         │      │ vs. accuracy, & model performance diff  │
└─────────────────────────────────────────┴──────┴─────────────────────────────────────────┘
```

---

## Component-by-Component Implementation Plan

### 1. Dependencies & Baseline Setup
#### [MODIFY] [requirements.txt](file:///home/aryan/ml_flow/requirements.txt)
- Add `scipy` (for Kolmogorov-Smirnov test)
- Add `river` (for ADWIN streaming drift detector) or `evidently` (for Evidently AI drift reports)
- Add `streamlit` and `plotly` (for dashboard visualization)

---

### 2. Multi-Method Drift Detection Engine
#### [NEW] [src/drift.py](file:///home/aryan/ml_flow/src/drift.py)
Create a unified drift detection module comparing incoming production data against `chunk_1` (baseline reference dataset):
- **Kolmogorov-Smirnov (KS) Test:** Two-sample non-parametric test on continuous numerical features ($p\text{-value} < 0.05$ indicates drift).
- **Population Stability Index (PSI):** Feature binning technique ($\text{PSI} > 0.25$ indicates significant drift).
- **ADWIN (Adaptive Windowing):** Streaming drift detector from `river.drift` tracking distribution changes in model prediction outputs.
- **Method Comparison Benchmark:** Measure and log the **execution time (ms)** and **detection accuracy/sensitivity** for each method.

---

### 3. Closed-Loop Airflow Retraining DAG
#### [MODIFY] [dags/churn_pipeline.py](file:///home/aryan/ml_flow/dags/churn_pipeline.py)
Transform the DAG into a closed-loop automated pipeline:
- **`check_drift_task`:** Executes `src/drift.py` on the newest data chunk against baseline.
- **`BranchPythonOperator`:** If drift threshold is crossed $\rightarrow$ route to `retrain_model_task`; otherwise $\rightarrow$ skip retraining and log status.
- **`retrain_eval_promote`:** Retrains on combined historical + drifted data, evaluates on a holdout set, logs to MLflow, and updates the production model in MLflow Registry / KServe.

---

### 4. Streamlit Interactive Dashboard
#### [NEW] [src/dashboard.py](file:///home/aryan/ml_flow/src/dashboard.py)
Develop a Streamlit dashboard presenting:
1. **Drift Score per Feature & Method:** Interactive bar/line charts comparing KS test $p$-values, PSI scores, and ADWIN flags.
2. **Retraining-Trigger Logs:** Table of historical drift detection events and triggered Airflow DAG executions.
3. **Before vs. After Model Accuracy:** F1-score, Precision, Recall, and ROC-AUC comparison before and after automated retraining.
4. **Method Evaluation (Accuracy vs. Computational Cost):** Scatter plot / bar graph comparing detection speed (ms) vs. drift detection sensitivity for KS, PSI, and ADWIN.

---

## Proposed Project Structure (Final State)

```
/home/aryan/ml_flow/
├── data/
│   ├── raw/
│   ├── chunks/                   # chunk_1.csv (Reference Baseline), chunk_2-4 (Incoming Batches)
│   └── processed/
├── models/
├── src/
│   ├── ingest.py
│   ├── preprocess.py
│   ├── train.py
│   ├── serve.py
│   ├── drift.py                  # [NEW] Multi-method drift detector (KS, PSI, ADWIN)
│   └── dashboard.py              # [NEW] Streamlit UI (drift scores, cost vs. accuracy, trigger logs)
├── dags/
│   └── churn_pipeline.py         # [MODIFY] Airflow DAG with conditional drift branch
├── k8s/
│   └── inference_service.yaml
├── requirements.txt              # [MODIFY] Added scipy, river/evidently, streamlit, plotly
└── implementation_plan.md
```

---

## Verification Plan

### Automated Tests & Verification
1. **Drift Detector Unit Test:** Run `python src/drift.py` on `chunk_1` (baseline) vs `chunk_2` / corrupted data; verify KS, PSI, and ADWIN return valid scores and computational runtime.
2. **Closed-Loop Trigger Test:** Execute Airflow DAG with no drift $\rightarrow$ verify retraining task is skipped. Inject drift $\rightarrow$ verify Airflow triggers full retraining + MLflow model update.
3. **Streamlit UI Test:** Launch `streamlit run src/dashboard.py` and verify all visual charts (Drift Scores, Before/After Accuracy, Cost vs. Accuracy) load properly.
