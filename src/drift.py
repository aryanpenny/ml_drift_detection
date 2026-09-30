import os
import pandas as pd
from scipy.stats import ks_2samp
import time
import numpy as np

def detect_ks_drift(reference_df, current_df, threshold= 0.05):
    drift_results = {}

    num_cols = reference_df.select_dtypes(include=['int64', 'float64']).columns

    for col in num_cols:
        stat, p_values = ks_2samp(reference_df[col],current_df[col])
        is_drifted = p_values < threshold
        drift_results[col]= {
            "p_value": round(p_values, 4),
            "drift_detected": is_drifted
        }
    
    return drift_results

def calculate_psi(reference, current, num_bins=10):
    counts_ref, bin_edges = np.histogram(reference, bins=num_bins)
    counts_cur, _ = np.histogram(current, bins=bin_edges)
    
    pct_ref = counts_ref / len(reference) + 1e-4
    pct_cur = counts_cur / len(current) + 1e-4
    
    psi_value = np.sum((pct_cur - pct_ref) * np.log(pct_cur / pct_ref))
    return round(float(psi_value), 4)

def should_retrain(chunk_number):
    ref_df = pd.read_csv("data/chunks/chunk_1.csv")
    cur_path = f"data/chunks/chunk_{chunk_number}.csv"

    if not os.path.exists(cur_path):
        return False
    
    cur_df = pd.read_csv(cur_path)
    ks_results = detect_ks_drift(ref_df, cur_df)
    
    # Returns True if ANY feature has drifted
    return any(res["drift_detected"] for res in ks_results.values())

if __name__ == "__main__":
    retrain_flag = should_retrain(2)
    print(f"Should Retrain for Chunk 2?: {retrain_flag}")

    