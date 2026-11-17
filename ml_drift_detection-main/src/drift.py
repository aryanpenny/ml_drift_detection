import os
import time
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from river import drift as river_drift


def detect_ks_drift(reference_df, current_df, threshold=0.05):
    """
    Two-sample Kolmogorov-Smirnov test for numerical features.
    Tests null hypothesis that two distributions are identical.
    """
    start_time = time.time()
    drift_results = {}
    num_cols = reference_df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns

    for col in num_cols:
        if col not in current_df.columns:
            continue
        # Drop NaNs for valid statistical test
        ref_series = reference_df[col].dropna()
        cur_series = current_df[col].dropna()
        if len(ref_series) == 0 or len(cur_series) == 0:
            continue
        stat, p_val = ks_2samp(ref_series, cur_series)
        is_drifted = bool(p_val < threshold)
        drift_results[col] = {
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_val), 4),
            "threshold": float(threshold),
            "drift_detected": is_drifted,
            "feature_type": "numerical"
        }

    exec_time = time.time() - start_time
    return drift_results, round(exec_time, 4)


def calculate_psi(reference, current, num_bins=10):
    """
    Population Stability Index (PSI) calculation for a numerical series or array.
    """
    ref_clean = np.asarray(reference[~np.isnan(reference)] if hasattr(reference, 'isna') else reference)
    cur_clean = np.asarray(current[~np.isnan(current)] if hasattr(current, 'isna') else current)

    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return 0.0

    counts_ref, bin_edges = np.histogram(ref_clean, bins=num_bins)
    counts_cur, _ = np.histogram(cur_clean, bins=bin_edges)

    pct_ref = counts_ref / len(ref_clean) + 1e-4
    pct_cur = counts_cur / len(cur_clean) + 1e-4

    psi_val = np.sum((pct_cur - pct_ref) * np.log(pct_cur / pct_ref))
    return round(float(psi_val), 4)


def detect_psi_drift(reference_df, current_df, threshold=0.1, num_bins=10):
    """
    Calculate PSI for numerical features across reference and current datasets.
    Severity scale:
      - PSI < 0.1: No Drift (Stable)
      - 0.1 <= PSI < 0.2: Moderate Drift (Slight change)
      - PSI >= 0.2: High Drift (Significant change)
    """
    start_time = time.time()
    psi_results = {}
    num_cols = reference_df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns

    for col in num_cols:
        if col not in current_df.columns:
            continue
        ref_series = reference_df[col].dropna().values
        cur_series = current_df[col].dropna().values
        if len(ref_series) == 0 or len(cur_series) == 0:
            continue

        psi_val = calculate_psi(ref_series, cur_series, num_bins=num_bins)
        is_drifted = bool(psi_val >= threshold)

        if psi_val < 0.1:
            severity = "No Drift"
        elif psi_val < 0.2:
            severity = "Moderate Drift"
        else:
            severity = "High Drift"

        psi_results[col] = {
            "psi_value": psi_val,
            "threshold": float(threshold),
            "severity": severity,
            "drift_detected": is_drifted,
            "feature_type": "numerical"
        }

    exec_time = time.time() - start_time
    return psi_results, round(exec_time, 4)


class FeatureADWINDetector:
    """
    Adaptive Windowing (ADWIN) drift detector for streaming / sequential data.
    Maintains separate ADWIN instances per numerical feature with baseline standardization.
    """
    def __init__(self, delta=0.002):
        self.delta = float(delta)
        self.detectors = {}
        self.stats = {}
        self.history = {}

    def fit_baseline(self, reference_df):
        """Fit normalization parameters (mean and std) from reference baseline data."""
        num_cols = reference_df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
        self.stats = {}
        self.detectors = {}
        self.history = {}

        for col in num_cols:
            series = reference_df[col].dropna()
            mean = float(series.mean())
            std = float(series.std()) if float(series.std()) > 0 else 1.0
            self.stats[col] = {"mean": mean, "std": std}
            self.detectors[col] = river_drift.ADWIN(delta=self.delta)
            self.history[col] = {
                "observations_processed": 0,
                "drift_events": [],
                "drift_detected": False,
                "last_detection_idx": None,
                "window_width": 0
            }

    def reset(self):
        """Reset all detector instances while preserving baseline stats."""
        for col in self.stats:
            self.detectors[col] = river_drift.ADWIN(delta=self.delta)
            self.history[col] = {
                "observations_processed": 0,
                "drift_events": [],
                "drift_detected": False,
                "last_detection_idx": None,
                "window_width": 0
            }

    def update_stream(self, df):
        """
        Stream observations sequentially through ADWIN detectors.
        Standardizes values using baseline mean and standard deviation.
        """
        start_time = time.time()
        for idx, row in df.iterrows():
            for col, detector in self.detectors.items():
                if col not in row or pd.isna(row[col]):
                    continue
                raw_val = float(row[col])
                mean = self.stats[col]["mean"]
                std = self.stats[col]["std"]
                norm_val = (raw_val - mean) / std

                detector.update(norm_val)
                self.history[col]["observations_processed"] += 1
                self.history[col]["window_width"] = getattr(detector, 'width', 0)

                if detector.drift_detected:
                    self.history[col]["drift_detected"] = True
                    self.history[col]["last_detection_idx"] = int(idx) if isinstance(idx, (int, np.integer)) else self.history[col]["observations_processed"]
                    self.history[col]["drift_events"].append({
                        "observation_index": self.history[col]["observations_processed"],
                        "window_width": getattr(detector, 'width', 0),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })

        exec_time = time.time() - start_time
        return self.get_summary(), round(exec_time, 4)

    def get_summary(self):
        """Get structured status summary per feature."""
        summary = {}
        for col, hist in self.history.items():
            summary[col] = {
                "drift_detected": hist["drift_detected"],
                "drift_count": len(hist["drift_events"]),
                "observations_processed": hist["observations_processed"],
                "last_detection_idx": hist["last_detection_idx"],
                "window_width": hist["window_width"],
                "delta": self.delta,
                "drift_events": hist["drift_events"]
            }
        return summary


def run_adwin_detection(reference_df, current_df, delta=0.002):
    """Convenience helper to run ADWIN across reference and current datasets."""
    detector = FeatureADWINDetector(delta=delta)
    detector.fit_baseline(reference_df)
    summary, exec_time = detector.update_stream(current_df)
    return summary, exec_time


def should_retrain(chunk_number, threshold=0.05, chunk_dir="data/chunks"):
    """
    Checks if retraining is required based on KS test drift detection.
    Preserves exact backward compatibility with existing pipeline and Airflow DAG.
    """
    ref_path = os.path.join(chunk_dir, "chunk_1.csv")
    cur_path = os.path.join(chunk_dir, f"chunk_{chunk_number}.csv")

    if not os.path.exists(cur_path) or not os.path.exists(ref_path):
        return False

    ref_df = pd.read_csv(ref_path)
    cur_df = pd.read_csv(cur_path)
    ks_results, _ = detect_ks_drift(ref_df, cur_df, threshold=threshold)

    return any(res["drift_detected"] for res in ks_results.values())


if __name__ == "__main__":
    print("Testing src/drift.py standalone...")
    chunk_1_path = "data/chunks/chunk_1.csv"
    chunk_2_path = "data/chunks/chunk_2.csv"
    if os.path.exists(chunk_1_path) and os.path.exists(chunk_2_path):
        ref = pd.read_csv(chunk_1_path)
        cur = pd.read_csv(chunk_2_path)
        ks, t_ks = detect_ks_drift(ref, cur)
        psi, t_psi = detect_psi_drift(ref, cur)
        adw, t_adw = run_adwin_detection(ref, cur)
        print(f"KS results (features: {len(ks)}, time: {t_ks}s)")
        print(f"PSI results (features: {len(psi)}, time: {t_psi}s)")
        print(f"ADWIN results (features: {len(adw)}, time: {t_adw}s)")
        print(f"Should Retrain for Chunk 2?: {should_retrain(2)}")
    else:
        print("Data chunks not found for standalone test.")
