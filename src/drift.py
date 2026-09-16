import os
import pandas as pd
from scipy.stats import ks_2samp
import time

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

if __name__ == "__main__":
    # Test loading chunk 1 (reference) and chunk 2 (current batch)
    ref_df = pd.read_csv("data/chunks/chunk_1.csv")
    cur_df = pd.read_csv("data/chunks/chunk_2.csv")
    
    start_time = time.time()
    results = detect_ks_drift(ref_df, cur_df)
    elapsed_time = (time.time() - start_time) * 1000
    
    print(f"KS Drift Detection completed in {elapsed_time:.2f} ms")
    print("Results:", results)