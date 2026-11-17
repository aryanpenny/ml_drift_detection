"""
Benchmark utility for Drift Detection methods:
- Kolmogorov-Smirnov (KS) Test
- Population Stability Index (PSI)
- Adaptive Windowing (ADWIN)

Generates controlled experimental scenarios to evaluate:
- Detection capability
- Detection rate
- False positive rate (under stable null scenario)
- Detection latency (first observation index where drift is detected)
- Execution time
"""

import os
import time
import numpy as np
import pandas as pd
from src.drift import detect_ks_drift, detect_psi_drift, run_adwin_detection


def generate_scenario(ref_df, scenario_type="no_drift", target_col="MonthlyCharges", severity=1.0):
    """
    Generate controlled experimental copies of reference data without mutating reference:
    - no_drift: identical distribution (bootstrap sample)
    - sudden_drift: sudden level shift in target feature
    - gradual_drift: linearly increasing drift trend across stream
    - variance_drift: shift in variance/dispersion
    """
    test_df = ref_df.copy()
    n = len(test_df)

    if scenario_type == "no_drift":
        # Reshuffle or bootstrap sample from same distribution
        test_df = ref_df.sample(n=n, replace=True, random_state=42).reset_index(drop=True)

    elif scenario_type == "sudden_drift":
        # Sudden shift from midpoint onwards
        half_idx = n // 2
        std = ref_df[target_col].std()
        shift = std * 2.0 * severity
        test_df.loc[half_idx:, target_col] = test_df.loc[half_idx:, target_col] + shift

    elif scenario_type == "gradual_drift":
        # Gradual linear increase from start to end
        std = ref_df[target_col].std()
        ramp = np.linspace(0, std * 2.5 * severity, n)
        test_df[target_col] = test_df[target_col] + ramp

    elif scenario_type == "variance_drift":
        # Increased spread/dispersion
        mean = ref_df[target_col].mean()
        test_df[target_col] = mean + (test_df[target_col] - mean) * (2.5 * severity)

    return test_df


def run_benchmark_suite(ref_df, ks_threshold=0.05, psi_threshold=0.1, adwin_delta=0.002):
    """
    Runs all 3 algorithms across 4 standard controlled scenarios.
    Returns structured results with latency, execution time, and detection flags.
    """
    scenarios = [
        {"name": "Stable Baseline (No Drift)", "type": "no_drift", "ground_truth": False},
        {"name": "Sudden Distribution Shift", "type": "sudden_drift", "ground_truth": True},
        {"name": "Gradual Linear Drift", "type": "gradual_drift", "ground_truth": True},
        {"name": "Variance Expansion", "type": "variance_drift", "ground_truth": True}
    ]

    target_col = "MonthlyCharges" if "MonthlyCharges" in ref_df.columns else ref_df.select_dtypes(include=[np.number]).columns[0]
    results = []

    for sc in scenarios:
        current_data = generate_scenario(ref_df, scenario_type=sc["type"], target_col=target_col)
        gt = sc["ground_truth"]

        # 1. Run KS
        ks_res, ks_time = detect_ks_drift(ref_df, current_data, threshold=ks_threshold)
        ks_detected = ks_res.get(target_col, {}).get("drift_detected", False)
        ks_p_val = ks_res.get(target_col, {}).get("p_value", None)
        ks_stat = ks_res.get(target_col, {}).get("statistic", None)

        # 2. Run PSI
        psi_res, psi_time = detect_psi_drift(ref_df, current_data, threshold=psi_threshold)
        psi_detected = psi_res.get(target_col, {}).get("drift_detected", False)
        psi_val = psi_res.get(target_col, {}).get("psi_value", None)
        psi_sev = psi_res.get(target_col, {}).get("severity", "None")

        # 3. Run ADWIN
        adwin_res, adwin_time = run_adwin_detection(ref_df, current_data, delta=adwin_delta)
        adw_info = adwin_res.get(target_col, {})
        adwin_detected = adw_info.get("drift_detected", False)
        adwin_count = adw_info.get("drift_count", 0)
        adwin_latency = adw_info.get("last_detection_idx", None)

        results.append({
            "scenario": sc["name"],
            "ground_truth": gt,
            "target_feature": target_col,
            "sample_size": len(current_data),
            # KS
            "ks_detected": ks_detected,
            "ks_statistic": ks_stat,
            "ks_p_value": ks_p_val,
            "ks_time_sec": ks_time,
            "ks_fp": (ks_detected and not gt),
            # PSI
            "psi_detected": psi_detected,
            "psi_value": psi_val,
            "psi_severity": psi_sev,
            "psi_time_sec": psi_time,
            "psi_fp": (psi_detected and not gt),
            # ADWIN
            "adwin_detected": adwin_detected,
            "adwin_detections": adwin_count,
            "adwin_latency": adwin_latency,
            "adwin_time_sec": adwin_time,
            "adwin_fp": (adwin_detected and not gt),
        })

    return pd.DataFrame(results)


def benchmark_data_volume(ref_df, volumes=[250, 500, 1000, 1700]):
    """
    Benchmarks execution time and detection under varying sample volumes.
    """
    target_col = "MonthlyCharges" if "MonthlyCharges" in ref_df.columns else ref_df.select_dtypes(include=[np.number]).columns[0]
    records = []

    for vol in volumes:
        sub_ref = ref_df.iloc[:vol].copy()
        sub_cur = generate_scenario(sub_ref, scenario_type="sudden_drift", target_col=target_col)

        _, t_ks = detect_ks_drift(sub_ref, sub_cur)
        _, t_psi = detect_psi_drift(sub_ref, sub_cur)
        _, t_adw = run_adwin_detection(sub_ref, sub_cur)

        records.append({
            "sample_volume": vol,
            "KS_time_ms": round(t_ks * 1000, 2),
            "PSI_time_ms": round(t_psi * 1000, 2),
            "ADWIN_time_ms": round(t_adw * 1000, 2)
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    import sys
    print("Testing src/benchmark.py...")
    ref_path = "data/chunks/chunk_1.csv"
    if os.path.exists(ref_path):
        ref_df = pd.read_csv(ref_path)
        bench = run_benchmark_suite(ref_df)
        print("Benchmark Results:")
        print(bench[["scenario", "ground_truth", "ks_detected", "psi_detected", "adwin_detected"]])
    else:
        print("Reference chunk not found.")
