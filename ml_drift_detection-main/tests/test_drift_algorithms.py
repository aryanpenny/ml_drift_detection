import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.drift import detect_ks_drift, detect_psi_drift, FeatureADWINDetector, run_adwin_detection, should_retrain


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 1000
    ref_df = pd.DataFrame({
        "MonthlyCharges": np.random.normal(65, 15, n),
        "TotalCharges": np.random.normal(2000, 500, n),
        "tenure": np.random.randint(1, 72, n)
    })
    return ref_df


def test_ks_no_drift(sample_data):
    np.random.seed(99)
    current_df = sample_data.sample(n=len(sample_data), replace=True).reset_index(drop=True)
    results, exec_time = detect_ks_drift(sample_data, current_df, threshold=0.01)
    
    for feat, res in results.items():
        assert "p_value" in res
        assert "drift_detected" in res
    assert exec_time >= 0


def test_ks_sudden_drift(sample_data):
    drifted_df = sample_data.copy()
    drifted_df["MonthlyCharges"] += 50.0
    results, _ = detect_ks_drift(sample_data, drifted_df, threshold=0.05)
    assert results["MonthlyCharges"]["drift_detected"] is True
    assert results["MonthlyCharges"]["p_value"] < 0.05


def test_psi_no_drift(sample_data):
    current_df = sample_data.sample(n=len(sample_data), replace=True).reset_index(drop=True)
    results, _ = detect_psi_drift(sample_data, current_df, threshold=0.1)
    assert results["MonthlyCharges"]["psi_value"] < 0.1
    assert results["MonthlyCharges"]["severity"] == "No Drift"
    assert results["MonthlyCharges"]["drift_detected"] is False


def test_psi_high_drift(sample_data):
    drifted_df = sample_data.copy()
    drifted_df["MonthlyCharges"] += 60.0
    results, _ = detect_psi_drift(sample_data, drifted_df, threshold=0.1)
    assert results["MonthlyCharges"]["psi_value"] >= 0.2
    assert results["MonthlyCharges"]["severity"] == "High Drift"
    assert results["MonthlyCharges"]["drift_detected"] is True


def test_adwin_stable_stream(sample_data):
    detector = FeatureADWINDetector(delta=0.002)
    detector.fit_baseline(sample_data)
    
    stable_stream = sample_data.sample(n=500, replace=True, random_state=123).reset_index(drop=True)
    summary, _ = detector.update_stream(stable_stream)
    
    assert summary["MonthlyCharges"]["drift_detected"] is False
    assert summary["MonthlyCharges"]["observations_processed"] == 500


def test_adwin_sudden_drift(sample_data):
    detector = FeatureADWINDetector(delta=0.002)
    detector.fit_baseline(sample_data)
    
    stream_df = sample_data.sample(n=400, replace=True, random_state=42).reset_index(drop=True)
    stream_df.loc[200:, "MonthlyCharges"] += 70.0
    
    summary, _ = detector.update_stream(stream_df)
    assert summary["MonthlyCharges"]["drift_detected"] is True
    assert summary["MonthlyCharges"]["drift_count"] >= 1
    assert summary["MonthlyCharges"]["last_detection_idx"] is not None


def test_adwin_detector_reset(sample_data):
    detector = FeatureADWINDetector(delta=0.002)
    detector.fit_baseline(sample_data)
    
    # Stream with sudden shift in middle
    stream_df = sample_data.sample(n=400, replace=True, random_state=42).reset_index(drop=True)
    stream_df.loc[200:, "MonthlyCharges"] += 80.0
    detector.update_stream(stream_df)
    
    assert detector.history["MonthlyCharges"]["drift_detected"] is True
    
    # Reset detector
    detector.reset()
    assert detector.history["MonthlyCharges"]["drift_detected"] is False
    assert detector.history["MonthlyCharges"]["observations_processed"] == 0
    assert len(detector.history["MonthlyCharges"]["drift_events"]) == 0


def test_missing_values_handling(sample_data):
    current_df = sample_data.copy()
    current_df.loc[:10, "MonthlyCharges"] = np.nan
    ks_res, _ = detect_ks_drift(sample_data, current_df)
    psi_res, _ = detect_psi_drift(sample_data, current_df)
    adw_res, _ = run_adwin_detection(sample_data, current_df)
    
    assert "MonthlyCharges" in ks_res
    assert "MonthlyCharges" in psi_res
    assert "MonthlyCharges" in adw_res
