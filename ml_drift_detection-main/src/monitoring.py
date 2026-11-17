"""
Monitoring and Pipeline Execution History Manager.
Persists execution logs, drift events, model metadata, and pipeline statuses to local JSON store.
"""

import os
import json
import time
from datetime import datetime

STATE_FILE = "data/monitoring_state.json"


def _load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "pipeline_history": [],
        "drift_events": [],
        "models": [],
        "active_model_version": "v1.0.0",
        "last_settings": {
            "ks_threshold": 0.05,
            "psi_threshold": 0.1,
            "adwin_delta": 0.002,
            "selected_chunk": 2
        }
    }


def _save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def record_drift_event(chunk_name, drifted_features, ks_res=None, psi_res=None, adwin_res=None):
    state = _load_state()
    event = {
        "id": f"drift_{int(time.time())}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "chunk": chunk_name,
        "drifted_features": drifted_features,
        "features_count": len(drifted_features),
        "ks_summary": {k: v["drift_detected"] for k, v in (ks_res or {}).items()},
        "psi_summary": {k: v["drift_detected"] for k, v in (psi_res or {}).items()},
        "adwin_summary": {k: v["drift_detected"] for k, v in (adwin_res or {}).items()}
    }
    state["drift_events"].insert(0, event)
    # keep last 50
    state["drift_events"] = state["drift_events"][:50]
    _save_state(state)
    return event


def record_pipeline_run(stage, status, duration_sec, details=None, trigger_reason="Manual"):
    state = _load_state()
    run = {
        "id": f"run_{int(time.time() * 1000)}",
        "stage": stage,
        "status": status,  # "Completed", "Running", "Failed", "Pending"
        "duration_sec": round(duration_sec, 2),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trigger_reason": trigger_reason,
        "details": details or {}
    }
    state["pipeline_history"].insert(0, run)
    state["pipeline_history"] = state["pipeline_history"][:50]
    _save_state(state)
    return run


def record_model_version(version, accuracy, precision, recall, f1, n_samples, trigger="Initial Training"):
    state = _load_state()
    model_record = {
        "version": version,
        "training_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "samples": n_samples,
        "trigger": trigger,
        "is_active": True
    }
    for m in state["models"]:
        m["is_active"] = False
    state["models"].insert(0, model_record)
    state["active_model_version"] = version
    _save_state(state)
    return model_record


def get_monitoring_state():
    return _load_state()


def save_settings(ks_thresh, psi_thresh, adwin_delta, chunk_num):
    state = _load_state()
    state["last_settings"] = {
        "ks_threshold": float(ks_thresh),
        "psi_threshold": float(psi_thresh),
        "adwin_delta": float(adwin_delta),
        "selected_chunk": int(chunk_num)
    }
    _save_state(state)
