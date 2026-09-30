import streamlit as st
import pandas as pd
import plotly.express as px
import sys

sys.path.insert(0, "/home/aryan/ml_flow")
from src.drift import detect_ks_drift

# Page setup
st.title("Churn Drift Dashboard")

# Load data
ref_df = pd.read_csv("data/chunks/chunk_1.csv")
cur_df = pd.read_csv("data/chunks/chunk_2.csv")

# Run drift detection
ks_results = detect_ks_drift(ref_df, cur_df)

# Convert to DataFrame for display
drift_df = pd.DataFrame(ks_results).T.reset_index()
drift_df.columns = ["Feature", "p_value", "drift_detected"]

# Show table
st.subheader("Drift Results")
st.dataframe(drift_df)

# Show bar chart
fig = px.bar(drift_df, x="Feature", y="p_value", color="drift_detected",
             title="KS Test p-values (below 0.05 = drift)")
st.plotly_chart(fig)
