# ML Drift Detection — Full Implementation Plan

## What This Project Does

This project builds a **complete Machine Learning pipeline** that:
1. Ingests and versions data
2. Trains a churn prediction model
3. Detects when incoming data has changed (drifted)
4. Automatically retrains the model when drift is found
5. Serves predictions via an API
6. Shows everything on a live dashboard

---

## ✅ What's Already Built (Completed)

| # | Component | File | Status |
|---|-----------|------|--------|
| 1 | Data ingestion & chunking | [src/ingest.py](file:///home/aryan/ml_flow/src/ingest.py) | ✅ Done |
| 2 | Data preprocessing & splitting | [src/preprocess.py](file:///home/aryan/ml_flow/src/preprocess.py) | ✅ Done |
| 3 | Model training with MLflow tracking | [src/train.py](file:///home/aryan/ml_flow/src/train.py) | ✅ Done |
| 4 | Data Drift detection (KS Test + PSI) | [src/drift.py](file:///home/aryan/ml_flow/src/drift.py) | ✅ Done |
| 5 | Closed-loop Airflow DAG (BranchPythonOperator) | [dags/churn_pipeline.py](file:///home/aryan/ml_flow/dags/churn_pipeline.py) | ✅ Done |
| 6 | FastAPI model serving (/predict endpoint) | [src/serve.py](file:///home/aryan/ml_flow/src/serve.py) | ✅ Done |
| 7 | Basic Streamlit dashboard (drift chart) | [src/dashboard.py](file:///home/aryan/ml_flow/src/dashboard.py) | ✅ Done |

---

## 🚀 Upgrade Plan: Multi-Drift Detection + Deep Learning

### The 3 Types of Drift We Will Detect

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  DATA DRIFT         LABEL DRIFT          CONCEPT DRIFT              │
│  "Questions          "Pass/fail ratio     "Answer key                │
│   changed"            changed"             changed"                  │
│                                                                      │
│  Feature             Target variable      Relationship between       │
│  distributions       distribution         features & target          │
│  shifted             shifted              changed                    │
│                                                                      │
│  Example:            Example:             Example:                   │
│  MonthlyCharges      Churn rate went      High-tenure customers      │
│  values increased    from 26% to 45%      now churn (they didn't     │
│                                           before)                    │
│                                                                      │
│  Detection:          Detection:           Detection:                 │
│  KS Test, PSI        Chi-squared test,    Model accuracy decay,      │
│  (Already done ✅)    proportion compare   Autoencoder error          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Phase 1: Add Label Drift Detection ⭐ Easy
#### [MODIFY] [src/drift.py](file:///home/aryan/ml_flow/src/drift.py)

**What it does:** Compares the percentage of churners in the baseline vs new data. If the churn rate jumps significantly, label drift is detected.

**What to add:**
- A `detect_label_drift(reference_df, current_df, target_col="Churn")` function
- Compare the proportion of `Churn=1` in both datasets
- Use a simple threshold (e.g., if difference > 10%) or a Chi-squared test from `scipy.stats`
- Return `{"baseline_rate": 0.26, "current_rate": 0.45, "label_drift_detected": True}`

**Why it matters:** If suddenly 45% of customers are churning instead of 26%, your model's predictions become unreliable even if the input features haven't changed.

---

### Phase 2: Add Concept Drift Detection ⭐⭐ Medium
#### [MODIFY] [src/drift.py](file:///home/aryan/ml_flow/src/drift.py)

**What it does:** Checks if the trained model's accuracy has dropped on new data. If the model was 82% accurate before but only 65% on new data, the "concept" (relationship between features and churn) has changed.

**What to add:**
- A `detect_concept_drift(model_path, new_data_path, accuracy_threshold=0.70)` function
- Load the saved model (`models/churn_model.pkl`)
- Load and preprocess the new chunk
- Run predictions on the new chunk and compare against actual labels
- If accuracy drops below the threshold → concept drift detected
- Return `{"baseline_accuracy": 0.82, "current_accuracy": 0.65, "concept_drift_detected": True}`

**Why it matters:** Data drift tells you the inputs changed. Concept drift tells you the model is actually making wrong predictions — which is what you really care about.

---

### Phase 3: Add Deep Learning Autoencoder ⭐⭐⭐ Advanced
#### [NEW] `src/autoencoder.py`

**What it does:** Trains a neural network to "compress and reconstruct" baseline data. When new data looks different from what the autoencoder learned, reconstruction error spikes — indicating drift.

**What to add:**
- A simple feedforward autoencoder using PyTorch or TensorFlow/Keras
- Architecture: `input(19 features) → encoder(10) → bottleneck(5) → decoder(10) → output(19 features)`
- Train on baseline `chunk_1` processed features
- For drift detection: pass new chunk through the autoencoder and measure Mean Squared Error (MSE)
- If MSE > threshold → drift detected (the autoencoder can't reconstruct data it hasn't seen before)

**New dependency to add to requirements.txt:**
```
torch>=2.0.0
```

**Why it's powerful:** KS test checks features one at a time. The autoencoder checks ALL features together and catches complex multi-feature drift patterns that statistical tests miss.

---

### Phase 4: Upgrade the Streamlit Dashboard ⭐⭐ Medium
#### [MODIFY] [src/dashboard.py](file:///home/aryan/ml_flow/src/dashboard.py)

**What to add (4 sections):**

1. **Data Drift Panel** (already exists — enhance it)
   - KS Test p-values per feature (bar chart) ✅ Already done
   - PSI scores per feature (bar chart)
   - Side-by-side feature distribution plots (histograms)

2. **Label Drift Panel** (new)
   - Baseline vs Current churn rate comparison (gauge chart or big numbers)
   - Chi-squared test result

3. **Concept Drift Panel** (new)
   - Model accuracy on baseline vs each new chunk (line chart over time)
   - Alert when accuracy drops below threshold

4. **Autoencoder Drift Panel** (new)
   - Reconstruction error distribution (histogram)
   - Threshold line showing when error = drift

5. **Method Comparison Table** (new)
   - Table comparing all detection methods: name, execution time (ms), drift detected (yes/no)
   - Shows which method is fastest vs most sensitive

---

### Phase 5: Upgrade Airflow DAG ⭐⭐ Medium
#### [MODIFY] [dags/churn_pipeline.py](file:///home/aryan/ml_flow/dags/churn_pipeline.py)

**What to change:**
- Update `check_drift()` function to run ALL drift checks (data + label + concept)
- Add a `drift_type` output so the dashboard knows which type triggered retraining
- Optionally add an `autoencoder_check_task` before the branch decision

---

### Phase 6: KServe Deployment ⭐⭐⭐ Advanced
#### [NEW] `k8s/inference_service.yaml` + `Dockerfile`

**Prerequisites:** Docker, Minikube, kubectl (all installed ✅)

**What to build:**
1. **Dockerfile** — Containerize your FastAPI `serve.py` + model into a Docker image
2. **InferenceService YAML** — Tell KServe to deploy your container on Kubernetes
3. **Deploy & Test** — Push the image, apply the YAML, hit the prediction endpoint

**Steps:**
1. Install Cert-Manager and KServe on Minikube
2. Write a `Dockerfile` that copies `src/serve.py`, `models/churn_model.pkl`, and installs dependencies
3. Build the Docker image: `docker build -t churn-model:v1 .`
4. Load into Minikube: `minikube image load churn-model:v1`
5. Write `k8s/inference_service.yaml` with the InferenceService spec
6. Deploy: `kubectl apply -f k8s/inference_service.yaml`
7. Test: `curl` the KServe endpoint with customer data

---

## Final Project Structure

```
/home/aryan/ml_flow/
├── data/
│   ├── raw/
│   ├── chunks/                   # chunk_1 (baseline), chunk_2-4 (incoming)
│   └── processed/
├── models/
│   └── churn_model.pkl
├── src/
│   ├── ingest.py                 # Data ingestion & chunking
│   ├── preprocess.py             # Data cleaning & train/test split
│   ├── train.py                  # Model training + MLflow logging
│   ├── serve.py                  # FastAPI prediction API
│   ├── drift.py                  # [UPGRADE] Data + Label + Concept drift
│   ├── autoencoder.py            # [NEW] Deep learning drift detector
│   └── dashboard.py              # [UPGRADE] Multi-panel Streamlit dashboard
├── dags/
│   └── churn_pipeline.py         # [UPGRADE] Multi-drift aware Airflow DAG
├── k8s/
│   ├── Dockerfile                # [NEW] Container for model serving
│   └── inference_service.yaml    # [NEW] KServe deployment spec
├── requirements.txt
└── implementation_plan.md
```

---

## Recommended Build Order

| Order | What to Build | Difficulty | Time Estimate |
|-------|--------------|------------|---------------|
| 1 | Label Drift in `drift.py` | ⭐ Easy | 30 min |
| 2 | Concept Drift in `drift.py` | ⭐⭐ Medium | 1 hour |
| 3 | Upgrade `dashboard.py` with all panels | ⭐⭐ Medium | 1-2 hours |
| 4 | Upgrade `churn_pipeline.py` for multi-drift | ⭐⭐ Medium | 30 min |
| 5 | Autoencoder in `autoencoder.py` | ⭐⭐⭐ Advanced | 2-3 hours |
| 6 | KServe (Dockerfile + YAML + deploy) | ⭐⭐⭐ Advanced | 2-3 hours |

---

## Verification Checklist

- [ ] `python src/drift.py` → Shows data drift, label drift, and concept drift results
- [ ] `python src/autoencoder.py` → Trains autoencoder, shows reconstruction error on new data
- [ ] `airflow dags test` → DAG checks all drift types, branches correctly
- [ ] `streamlit run src/dashboard.py` → All 4 panels render with charts
- [ ] `kubectl get inferenceservice` → KServe model is deployed and serving
- [ ] `curl <kserve-endpoint>` → Returns churn prediction from Kubernetes
