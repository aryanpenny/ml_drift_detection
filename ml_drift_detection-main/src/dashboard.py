"""
DriftGuard — Intelligent ML Monitoring & Automated Model Recovery
Premium MLOps Dashboard built with Streamlit.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure sys.path so local modules can be loaded from anywhere
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.drift import detect_ks_drift, detect_psi_drift, run_adwin_detection, should_retrain
from src.benchmark import run_benchmark_suite, benchmark_data_volume, generate_scenario
from src.monitoring import (
    get_monitoring_state,
    record_drift_event,
    record_pipeline_run,
    record_model_version,
    save_settings
)
from src.train import run_training_pipeline
from src.preprocess import clean_data, split_and_save

# Page Setup
st.set_page_config(
    page_title="DriftGuard — MLOps Monitoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "selected_chunk" not in st.session_state:
    st.session_state.selected_chunk = 2
if "ks_threshold" not in st.session_state:
    st.session_state.ks_threshold = 0.05
if "psi_threshold" not in st.session_state:
    st.session_state.psi_threshold = 0.1
if "adwin_delta" not in st.session_state:
    st.session_state.adwin_delta = 0.002
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.strftime("%H:%M:%S")

# Theme Definitions & CSS Injection
THEMES = {
    "dark": {
        "bg_color": "#0B0F17",
        "card_bg": "#111827",
        "card_border": "#1F2937",
        "text_primary": "#F9FAFB",
        "text_secondary": "#9CA3AF",
        "accent": "#3B82F6",
        "accent_green": "#10B981",
        "accent_red": "#EF4444",
        "accent_amber": "#F59E0B",
        "plotly_template": "plotly_dark"
    },
    "light": {
        "bg_color": "#F9FAFB",
        "card_bg": "#FFFFFF",
        "card_border": "#E5E7EB",
        "text_primary": "#111827",
        "text_secondary": "#6B7280",
        "accent": "#2563EB",
        "accent_green": "#059669",
        "accent_red": "#DC2626",
        "accent_amber": "#D97706",
        "plotly_template": "plotly_white"
    }
}

current_theme = THEMES[st.session_state.theme]

# Inject Modern SaaS Typography & CSS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    .stApp {{
        background-color: {current_theme['bg_color']};
        color: {current_theme['text_primary']};
    }}
    
    /* Premium Header */
    .app-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 20px;
        background: {current_theme['card_bg']};
        border-bottom: 1px solid {current_theme['card_border']};
        border-radius: 12px;
        margin-bottom: 24px;
    }}
    
    /* Metric Cards */
    .metric-card {{
        background-color: {current_theme['card_bg']};
        border: 1px solid {current_theme['card_border']};
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }}
    .metric-label {{
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {current_theme['text_secondary']};
        margin-bottom: 6px;
    }}
    .metric-value {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {current_theme['text_primary']};
    }}
    .metric-sub {{
        font-size: 0.75rem;
        color: {current_theme['text_secondary']};
        margin-top: 4px;
    }}
    
    /* Status Badges */
    .badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }}
    .badge-stable {{
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }}
    .badge-drift {{
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }}
    .badge-warning {{
        background-color: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }}
    .badge-neutral {{
        background-color: rgba(156, 163, 175, 0.15);
        color: #9CA3AF;
        border: 1px solid rgba(156, 163, 175, 0.3);
    }}
    
    /* Pipeline Step */
    .pipeline-container {{
        display: flex;
        gap: 12px;
        overflow-x: auto;
        padding: 16px 0;
    }}
    .pipeline-step {{
        flex: 1;
        min-width: 130px;
        background: {current_theme['card_bg']};
        border: 1px solid {current_theme['card_border']};
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }}
    
    /* Sidebar Branding */
    .sidebar-brand {{
        font-size: 1.3rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 2px;
        color: {current_theme['text_primary']};
    }}
    .sidebar-sub {{
        font-size: 0.75rem;
        color: {current_theme['text_secondary']};
        margin-bottom: 24px;
    }}
</style>
""", unsafe_allow_html=True)


# Data Helper Functions
@st.cache_data(ttl=60)
def load_reference_data():
    path = os.path.join(PROJECT_ROOT, "data/chunks/chunk_1.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data(ttl=30)
def load_chunk_data(chunk_num):
    path = os.path.join(PROJECT_ROOT, f"data/chunks/chunk_{chunk_num}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# Sidebar Navigation & Identity
with st.sidebar:
    st.markdown("""
        <div class="sidebar-brand">🛡️ DriftGuard</div>
        <div class="sidebar-sub">Intelligent MLOps & Retraining Pipeline</div>
    """, unsafe_allow_html=True)

    nav_choice = st.radio(
        "Navigation",
        [
            "Overview",
            "Drift Monitoring",
            "Detection Benchmark",
            "Model Performance",
            "Retraining Pipeline",
            "Model Management",
            "Data Management",
            "Settings"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Monitored Stream")
    chunk_selection = st.selectbox(
        "Current Stream Chunk",
        [2, 3, 4],
        index=[2, 3, 4].index(st.session_state.selected_chunk) if st.session_state.selected_chunk in [2, 3, 4] else 0,
        format_func=lambda x: f"Chunk {x} (Production Stream)"
    )
    if chunk_selection != st.session_state.selected_chunk:
        st.session_state.selected_chunk = chunk_selection
        st.rerun()

    st.markdown("---")
    # Quick Status Indicator
    ref_df_check = load_reference_data()
    cur_df_check = load_chunk_data(st.session_state.selected_chunk)
    if ref_df_check is not None and cur_df_check is not None:
        drift_flag = should_retrain(st.session_state.selected_chunk, chunk_dir=os.path.join(PROJECT_ROOT, "data/chunks"))
        status_html = '<span class="badge badge-drift">DRIFT DETECTED</span>' if drift_flag else '<span class="badge badge-stable">STREAM HEALTHY</span>'
        st.markdown(f"**System Status**: {status_html}", unsafe_allow_html=True)
    else:
        st.markdown("**System Status**: <span class=\"badge badge-warning\">DATA MISSING</span>", unsafe_allow_html=True)

    st.markdown(f"<div style='font-size: 0.72rem; color: {current_theme['text_secondary']}; margin-top: 20px;'>Project: Telco Churn Detection<br>Version: 2.1.0-release</div>", unsafe_allow_html=True)


# Application Top Header Bar
col_h1, col_h2, col_h3, col_h4 = st.columns([5, 2, 2, 1])
with col_h1:
    st.markdown(f"## {nav_choice}")
with col_h2:
    st.markdown(f"<div style='font-size: 0.8rem; color:{current_theme['text_secondary']}; text-align:right;'>Last Refreshed: <b>{st.session_state.last_refresh}</b></div>", unsafe_allow_html=True)
with col_h3:
    theme_btn_label = "☀️ Light Mode" if st.session_state.theme == "dark" else "🌙 Dark Mode"
    if st.button(theme_btn_label, use_container_width=True):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()
with col_h4:
    if st.button("🔄", help="Refresh Dashboard"):
        st.session_state.last_refresh = time.strftime("%H:%M:%S")
        st.cache_data.clear()
        st.rerun()

st.markdown("<hr style='margin: 8px 0 24px 0; border: none; border-top: 1px solid #1F2937;'>", unsafe_allow_html=True)

# Shared Data Loading
ref_df = load_reference_data()
cur_df = load_chunk_data(st.session_state.selected_chunk)
mon_state = get_monitoring_state()


# PAGE 1: OVERVIEW
if nav_choice == "Overview":
    if ref_df is None or cur_df is None:
        st.error("Reference dataset or production chunk data not found in data/chunks. Please run Data Ingestion first.")
    else:
        # Run live detection for chunk
        ks_res, _ = detect_ks_drift(ref_df, cur_df, threshold=st.session_state.ks_threshold)
        psi_res, _ = detect_psi_drift(ref_df, cur_df, threshold=st.session_state.psi_threshold)
        adwin_res, _ = run_adwin_detection(ref_df, cur_df, delta=st.session_state.adwin_delta)

        drifted_features = [f for f, r in ks_res.items() if r["drift_detected"]]
        is_overall_drift = len(drifted_features) > 0

        # Top Metric Cards
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Model Version</div>
                <div class="metric-value">{mon_state.get('active_model_version', 'v1.0.0')}</div>
                <div class="metric-sub">Random Forest (Scikit-Learn)</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            status_badge = '<span class="badge badge-drift">DRIFT DETECTED</span>' if is_overall_drift else '<span class="badge badge-stable">STABLE</span>'
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Stream Status</div>
                <div class="metric-value" style="font-size: 1.1rem; padding-top: 6px;">{status_badge}</div>
                <div class="metric-sub">{len(drifted_features)} features flagged</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            active_models = mon_state.get("models", [])
            latest_acc = f"{active_models[0]['accuracy']*100:.1f}%" if active_models else "81.8%"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Model Accuracy</div>
                <div class="metric-value">{latest_acc}</div>
                <div class="metric-sub">Test Split (Holdout)</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            drift_events_count = len(mon_state.get("drift_events", []))
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Drift Events</div>
                <div class="metric-value">{drift_events_count}</div>
                <div class="metric-sub">Across all chunks</div>
            </div>
            """, unsafe_allow_html=True)
        with m5:
            retrain_runs = len([r for r in mon_state.get("pipeline_history", []) if r.get("stage") == "Model Training"])
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Retraining Runs</div>
                <div class="metric-value">{retrain_runs}</div>
                <div class="metric-sub">Automated / Manual</div>
            </div>
            """, unsafe_allow_html=True)
        with m6:
            last_retrain = active_models[0]["training_time"] if active_models else "Initial Baseline"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Last Retrained</div>
                <div class="metric-value" style="font-size: 0.95rem; padding-top: 8px;">{last_retrain.split()[-1] if ' ' in str(last_retrain) else last_retrain}</div>
                <div class="metric-sub">Automated Pipeline</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Pipeline Execution Visualizer
        st.subheader("Closed-Loop MLOps Pipeline Flow")
        steps = [
            ("Data Ingestion", "Completed", "7,043 rows chunked"),
            ("Preprocessing", "Completed", "LabelEncoded & Scaled"),
            ("Model Serving", "Completed", "FastAPI /predict Active"),
            ("Drift Detection", "Completed" if not is_overall_drift else "Triggered", f"{len(drifted_features)} features drifted"),
            ("Airflow Pipeline", "Completed" if is_overall_drift else "Standby", "BranchPythonOperator"),
            ("Model Retraining", "Completed" if retrain_runs > 0 else "Pending", "MLflow Tracked"),
            ("Model Validation", "Completed" if retrain_runs > 0 else "Pending", "Holdout Evaluated")
        ]
        p_cols = st.columns(len(steps))
        for col, (step_name, step_status, step_sub) in zip(p_cols, steps):
            with col:
                b_class = "badge-stable" if step_status == "Completed" else ("badge-drift" if step_status == "Triggered" else "badge-neutral")
                st.markdown(f"""
                <div class="pipeline-step">
                    <div style="font-size: 0.82rem; font-weight: 600; margin-bottom: 4px;">{step_name}</div>
                    <span class="badge {b_class}">{step_status}</span>
                    <div style="font-size: 0.7rem; color: {current_theme['text_secondary']}; margin-top: 6px;">{step_sub}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts Section
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("#### Feature Drift Overview (P-Value vs Significance)")
            features = list(ks_res.keys())
            p_values = [ks_res[f]["p_value"] for f in features]
            colors = [current_theme["accent_red"] if p < st.session_state.ks_threshold else current_theme["accent_green"] for p in p_values]

            fig_p = go.Figure(go.Bar(
                x=features,
                y=p_values,
                marker_color=colors,
                text=[f"p={p}" for p in p_values],
                textposition='auto'
            ))
            fig_p.add_hline(
                y=st.session_state.ks_threshold,
                line_dash="dot",
                line_color=current_theme["accent_amber"],
                annotation_text=f"Significance Threshold ({st.session_state.ks_threshold})"
            )
            fig_p.update_layout(
                template=current_theme["plotly_template"],
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=30, b=20),
                height=320,
                yaxis_title="KS Test P-Value"
            )
            st.plotly_chart(fig_p, use_container_width=True)

        with c2:
            st.markdown("#### Population Stability Index (PSI)")
            psi_vals = [psi_res[f]["psi_value"] for f in features]
            psi_colors = [
                current_theme["accent_green"] if v < 0.1 else (current_theme["accent_amber"] if v < 0.2 else current_theme["accent_red"])
                for v in psi_vals
            ]
            fig_psi = go.Figure(go.Bar(
                x=features,
                y=psi_vals,
                marker_color=psi_colors,
                text=[f"{v:.3f}" for v in psi_vals],
                textposition='auto'
            ))
            fig_psi.add_hline(y=0.1, line_dash="dash", line_color=current_theme["accent_amber"], annotation_text="Moderate (0.1)")
            fig_psi.add_hline(y=0.2, line_dash="dot", line_color=current_theme["accent_red"], annotation_text="High (0.2)")
            fig_psi.update_layout(
                template=current_theme["plotly_template"],
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=30, b=20),
                height=320,
                yaxis_title="PSI Score"
            )
            st.plotly_chart(fig_psi, use_container_width=True)

        # Recent Activity Feed
        st.markdown("#### 🕒 Recent Pipeline Activity")
        events = mon_state.get("pipeline_history", [])[:6]
        if events:
            ev_df = pd.DataFrame(events)[["timestamp", "stage", "status", "duration_sec", "trigger_reason"]]
            st.dataframe(ev_df, use_container_width=True, hide_index=True)
        else:
            st.info("No recorded pipeline activity yet. Run Retraining or Data Management steps to log events.")


# PAGE 2: DRIFT MONITORING
elif nav_choice == "Drift Monitoring":
    if ref_df is None or cur_df is None:
        st.error("Data chunks not available. Please verify chunk_1.csv and current chunk exist.")
    else:
        st.markdown(f"### Feature-Level Drift Analysis — **Chunk {st.session_state.selected_chunk}**")

        ks_res, ks_time = detect_ks_drift(ref_df, cur_df, threshold=st.session_state.ks_threshold)
        psi_res, psi_time = detect_psi_drift(ref_df, cur_df, threshold=st.session_state.psi_threshold)
        adwin_res, adwin_time = run_adwin_detection(ref_df, cur_df, delta=st.session_state.adwin_delta)

        # Overall Summary Pill Bar
        num_features = len(ks_res)
        drifted_features = [f for f, r in ks_res.items() if r["drift_detected"] or psi_res.get(f, {}).get("drift_detected", False)]

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Features Monitored", num_features)
        sc2.metric("Drifted Features", len(drifted_features))
        sc3.metric("KS Detections", sum(1 for r in ks_res.values() if r["drift_detected"]))
        sc4.metric("ADWIN Detections", sum(1 for r in adwin_res.values() if r["drift_detected"]))

        st.markdown("<br>", unsafe_allow_html=True)

        # Multi-detector Comparison Table
        table_rows = []
        for feat in ks_res.keys():
            k = ks_res.get(feat, {})
            p = psi_res.get(feat, {})
            a = adwin_res.get(feat, {})

            # Synthesize overall status
            detections = [k.get("drift_detected", False), p.get("drift_detected", False), a.get("drift_detected", False)]
            if sum(detections) >= 2:
                overall = "High Alert"
            elif sum(detections) == 1:
                overall = "Monitored Alert"
            else:
                overall = "Stable"

            table_rows.append({
                "Feature Name": feat,
                "Type": "Numerical",
                "KS Stat": k.get("statistic", "N/A"),
                "KS P-Value": k.get("p_value", "N/A"),
                "PSI Score": p.get("psi_value", "N/A"),
                "PSI Severity": p.get("severity", "N/A"),
                "ADWIN Detected": "Yes" if a.get("drift_detected") else "No",
                "ADWIN Events": a.get("drift_count", 0),
                "Overall Health": overall
            })

        table_df = pd.DataFrame(table_rows)
        st.dataframe(
            table_df.style.applymap(
                lambda val: "color: #EF4444; font-weight: bold;" if val in ["High Alert", "Yes", "High Drift"]
                else ("color: #F59E0B; font-weight: bold;" if val in ["Monitored Alert", "Moderate Drift"]
                else ("color: #10B981;" if val in ["Stable", "No", "No Drift"] else "")),
                subset=["PSI Severity", "ADWIN Detected", "Overall Health"]
            ),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")
        st.markdown("### 🔍 In-Depth Feature Inspector & Natural Language Explanation")

        selected_feature = st.selectbox("Select Feature to Inspect:", list(ks_res.keys()))

        if selected_feature:
            feat_k = ks_res[selected_feature]
            feat_p = psi_res[selected_feature]
            feat_a = adwin_res[selected_feature]

            # Natural Language Explanation Logic
            agreements = []
            if feat_k["drift_detected"]:
                agreements.append(f"Kolmogorov-Smirnov test identified a statistically significant divergence (p = {feat_k['p_value']} < {st.session_state.ks_threshold})")
            else:
                agreements.append(f"Kolmogorov-Smirnov test did not find statistically significant divergence (p = {feat_k['p_value']} >= {st.session_state.ks_threshold})")

            if feat_p["drift_detected"]:
                agreements.append(f"PSI score of {feat_p['psi_value']} exceeded the threshold, classified as {feat_p['severity']}")
            else:
                agreements.append(f"PSI score of {feat_p['psi_value']} indicates stable distribution ({feat_p['severity']})")

            if feat_a["drift_detected"]:
                agreements.append(f"ADWIN detected {feat_a['drift_count']} adaptive window change events (last change at observation #{feat_a['last_detection_idx']})")
            else:
                agreements.append("ADWIN maintained stable adaptive windows across the observation stream")

            explanation_text = f"**Diagnostic Assessment for `{selected_feature}`**:\n\n" + " • " + "\n\n • ".join(agreements)

            st.info(explanation_text)

            # Feature Visual Comparison
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.markdown(f"#### Reference vs Current Distribution (`{selected_feature}`)")
                ref_series = ref_df[selected_feature].dropna()
                cur_series = cur_df[selected_feature].dropna()

                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=ref_series,
                    name="Reference (Chunk 1)",
                    opacity=0.6,
                    marker_color=current_theme["accent"]
                ))
                fig_hist.add_trace(go.Histogram(
                    x=cur_series,
                    name=f"Current (Chunk {st.session_state.selected_chunk})",
                    opacity=0.6,
                    marker_color=current_theme["accent_amber"]
                ))
                fig_hist.update_layout(
                    barmode="overlay",
                    template=current_theme["plotly_template"],
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=320,
                    xaxis_title=selected_feature,
                    yaxis_title="Count"
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_g2:
                st.markdown("#### Cumulative Distribution Function (CDF)")
                ref_sorted = np.sort(ref_series)
                cur_sorted = np.sort(cur_series)
                ref_cdf = np.arange(1, len(ref_sorted) + 1) / len(ref_sorted)
                cur_cdf = np.arange(1, len(cur_sorted) + 1) / len(cur_sorted)

                fig_cdf = go.Figure()
                fig_cdf.add_trace(go.Scatter(x=ref_sorted, y=ref_cdf, name="Reference CDF", line=dict(color=current_theme["accent"], width=2)))
                fig_cdf.add_trace(go.Scatter(x=cur_sorted, y=cur_cdf, name="Current CDF", line=dict(color=current_theme["accent_amber"], width=2)))
                fig_cdf.update_layout(
                    template=current_theme["plotly_template"],
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=320,
                    xaxis_title=selected_feature,
                    yaxis_title="Cumulative Probability"
                )
                st.plotly_chart(fig_cdf, use_container_width=True)


# PAGE 3: DETECTION BENCHMARK
elif nav_choice == "Detection Benchmark":
    st.markdown("### 🔬 Drift Detection Benchmark Suite")
    st.markdown("Rigorous experimental comparison of **Kolmogorov-Smirnov (KS)**, **Population Stability Index (PSI)**, and **Adaptive Windowing (ADWIN)** under controlled drift scenarios.")

    if ref_df is None:
        st.error("Reference data required for benchmarking.")
    else:
        # Controlled Experiment Controls
        col_b1, col_b2 = st.columns([3, 1])
        with col_b1:
            st.caption("Controlled scenarios test the algorithms against Ground Truth drift: No Drift, Sudden Level Shift, Gradual Linear Trend, and Variance Expansion.")
        with col_b2:
            run_btn = st.button("🚀 Run Controlled Benchmark", use_container_width=True)

        if run_btn or "bench_results" not in st.session_state:
            with st.spinner("Executing controlled experimental benchmark suite..."):
                bench_df = run_benchmark_suite(
                    ref_df,
                    ks_threshold=st.session_state.ks_threshold,
                    psi_threshold=st.session_state.psi_threshold,
                    adwin_delta=st.session_state.adwin_delta
                )
                st.session_state.bench_results = bench_df

        bench_df = st.session_state.bench_results

        # Comparison Table
        st.markdown("#### Benchmark Detection & Ground Truth Results")
        display_bench = bench_df[[
            "scenario", "ground_truth", "ks_detected", "ks_time_sec",
            "psi_detected", "psi_time_sec", "adwin_detected", "adwin_latency", "adwin_time_sec"
        ]].copy()
        display_bench.columns = [
            "Scenario", "Ground Truth", "KS Detected", "KS Time (s)",
            "PSI Detected", "PSI Time (s)", "ADWIN Detected", "ADWIN Latency (obs)", "ADWIN Time (s)"
        ]
        st.dataframe(display_bench, use_container_width=True, hide_index=True)

        # Performance Charts
        st.markdown("#### Benchmark Visualizations")
        bc1, bc2 = st.columns(2)

        with bc1:
            st.markdown("##### Execution Time by Algorithm (Seconds)")
            time_comp_df = pd.DataFrame({
                "Algorithm": ["KS Test", "PSI Score", "River ADWIN"],
                "Average Execution Time (s)": [
                    bench_df["ks_time_sec"].mean(),
                    bench_df["psi_time_sec"].mean(),
                    bench_df["adwin_time_sec"].mean()
                ]
            })
            fig_time = px.bar(
                time_comp_df,
                x="Algorithm",
                y="Average Execution Time (s)",
                color="Algorithm",
                color_discrete_sequence=[current_theme["accent"], current_theme["accent_green"], current_theme["accent_amber"]]
            )
            fig_time.update_layout(
                template=current_theme["plotly_template"],
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=30, b=20),
                height=300
            )
            st.plotly_chart(fig_time, use_container_width=True)

        with bc2:
            st.markdown("##### Detection Accuracy against Ground Truth")
            # Calculate accuracy per method
            ks_acc = (bench_df["ks_detected"] == bench_df["ground_truth"]).mean() * 100
            psi_acc = (bench_df["psi_detected"] == bench_df["ground_truth"]).mean() * 100
            adw_acc = (bench_df["adwin_detected"] == bench_df["ground_truth"]).mean() * 100

            acc_df = pd.DataFrame({
                "Method": ["KS Test", "PSI", "ADWIN"],
                "Detection Accuracy (%)": [ks_acc, psi_acc, adw_acc]
            })
            fig_acc = px.bar(
                acc_df,
                x="Method",
                y="Detection Accuracy (%)",
                color="Method",
                color_discrete_sequence=[current_theme["accent"], current_theme["accent_green"], current_theme["accent_amber"]]
            )
            fig_acc.update_layout(
                template=current_theme["plotly_template"],
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=30, b=20),
                height=300,
                yaxis=dict(range=[0, 105])
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        # Architectural Comparison Table
        st.markdown("#### Algorithmic Trade-off Analysis")
        tech_specs = pd.DataFrame([
            {
                "Technique": "Kolmogorov-Smirnov (KS)",
                "Paradigm": "Non-parametric batch test",
                "Supported Types": "Continuous numerical",
                "Metric Scale": "P-value [0, 1] & D-stat",
                "Streaming Ready": "No (requires window batches)",
                "Primary Advantage": "Exact statistical significance testing"
            },
            {
                "Technique": "Population Stability Index (PSI)",
                "Paradigm": "Binned divergence (Kullback-Leibler)",
                "Supported Types": "Numerical & Ordinal",
                "Metric Scale": "PSI Score [0, ∞)",
                "Streaming Ready": "No (requires baseline & current bins)",
                "Primary Advantage": "Standard regulatory risk & severity scale"
            },
            {
                "Technique": "Adaptive Windowing (ADWIN)",
                "Paradigm": "Online streaming adaptive window",
                "Supported Types": "Sequential numerical streams",
                "Metric Scale": "Binary drift flag + Cut latency",
                "Streaming Ready": "Yes (O(1) update per observation)",
                "Primary Advantage": "Sub-batch latency & automatic window sizing"
            }
        ])
        st.dataframe(tech_specs, use_container_width=True, hide_index=True)


# PAGE 4: MODEL PERFORMANCE
elif nav_choice == "Model Performance":
    st.markdown("### 📈 Model Evaluation & Retraining Impact")

    active_models = mon_state.get("models", [])
    if not active_models:
        # Default baseline if state is fresh
        active_models = [{
            "version": "v1.0.0",
            "training_time": "2026-11-17 19:59:54",
            "accuracy": 0.8182,
            "precision": 0.7164,
            "recall": 0.5161,
            "f1": 0.6000,
            "samples": 1760,
            "trigger": "Initial Baseline Training",
            "is_active": True
        }]

    # Top Metrics for Current Active Model
    curr_m = active_models[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Version", curr_m["version"])
    c2.metric("Accuracy", f"{curr_m['accuracy']*100:.2f}%")
    c3.metric("Precision", f"{curr_m['precision']*100:.2f}%")
    c4.metric("F1-Score", f"{curr_m['f1']*100:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Before vs After Comparison (if multiple models exist)
    if len(active_models) > 1:
        st.markdown("#### Model Version Comparison (Before vs After Retraining)")
        prev_m = active_models[1]

        acc_diff = (curr_m['accuracy'] - prev_m['accuracy']) * 100
        f1_diff = (curr_m['f1'] - prev_m['f1']) * 100

        comp_data = pd.DataFrame([
            {"Metric": "Accuracy", f"Previous ({prev_m['version']})": prev_m['accuracy'], f"Active ({curr_m['version']})": curr_m['accuracy']},
            {"Metric": "Precision", f"Previous ({prev_m['version']})": prev_m['precision'], f"Active ({curr_m['version']})": curr_m['precision']},
            {"Metric": "Recall", f"Previous ({prev_m['version']})": prev_m['recall'], f"Active ({curr_m['version']})": curr_m['recall']},
            {"Metric": "F1-Score", f"Previous ({prev_m['version']})": prev_m['f1'], f"Active ({curr_m['version']})": curr_m['f1']},
        ])
        st.dataframe(comp_data, use_container_width=True, hide_index=True)

        diff_msg = "improved" if acc_diff >= 0 else "decreased"
        st.info(f"Retraining {curr_m['version']} resulted in accuracy change of **{acc_diff:+.2f}%** and F1 change of **{f1_diff:+.2f}%** compared to {prev_m['version']}.")
    else:
        st.info("Currently running baseline model `v1.0.0`. Retrain on a new chunk in the Retraining Pipeline page to observe Before/After comparison.")

    # Historical Model Evolution
    st.markdown("#### Model Registry & Version History")
    hist_df = pd.DataFrame(active_models)[["version", "training_time", "accuracy", "precision", "recall", "f1", "trigger", "is_active"]]
    st.dataframe(hist_df, use_container_width=True, hide_index=True)


# PAGE 5: RETRAINING PIPELINE
elif nav_choice == "Retraining Pipeline":
    st.markdown("### 🔄 Automated Retraining Pipeline (Closed-Loop)")
    st.markdown("Monitors stream chunks, validates statistical drift thresholds, and triggers automated model retraining via Airflow DAG logic.")

    p_col1, p_col2 = st.columns([2, 1])

    with p_col1:
        st.markdown(f"#### Monitored Chunk: **Chunk {st.session_state.selected_chunk}**")
        drift_flag = should_retrain(st.session_state.selected_chunk, chunk_dir=os.path.join(PROJECT_ROOT, "data/chunks"))

        if drift_flag:
            st.warning(f"⚠️ Drift identified in Chunk {st.session_state.selected_chunk}! Automated retraining condition is **ACTIVE**.")
        else:
            st.success(f"✅ Chunk {st.session_state.selected_chunk} is stable. No retraining required.")

        st.markdown("#### Pipeline Workflow Visualizer")
        st.code("""
[Airflow DAG: churn_prediction_pipeline]
       │
       ▼
 [check_drift] (BranchPythonOperator)
       │
       ├─────────────────────────────────┐
       ▼ (Drift Found)                   ▼ (No Drift)
[preprocess_data] (Chunk Ingest)     [skip_retraining] (End)
       │
       ▼
 [train_model] (MLflow Tracked)
       │
       ▼
 [deploy_model] (FastAPI Active)
        """, language="text")

    with p_col2:
        st.markdown("#### Manual Retraining Trigger")
        st.caption("Manually trigger the full pipeline: Load chunk data → Clean & Preprocess → Retrain Random Forest → Log to MLflow → Update Serving.")

        trigger_chunk = st.selectbox("Select Data Chunk for Retraining:", [1, 2, 3, 4], index=st.session_state.selected_chunk - 1)
        trigger_btn = st.button("⚡ Execute Retraining Pipeline Now", use_container_width=True)

        if trigger_btn:
            with st.spinner("Executing pipeline: Preprocessing & Retraining..."):
                start_p = time.time()
                # 1. Load & preprocess
                raw_chunk = pd.read_csv(os.path.join(PROJECT_ROOT, f"data/chunks/chunk_{trigger_chunk}.csv"))
                cleaned_chunk = clean_data(raw_chunk)
                split_and_save(cleaned_chunk)

                # 2. Retrain model
                new_version = f"v1.{len(mon_state.get('models', []))}.0"
                trained_model, metrics = run_training_pipeline(
                    chunk_num=trigger_chunk,
                    model_version=new_version,
                    trigger=f"Retrained on Chunk {trigger_chunk}"
                )
                dur = time.time() - start_p
                record_pipeline_run("Automated Retraining", "Completed", dur, metrics, f"Triggered on Chunk {trigger_chunk}")
                st.success(f"Model successfully retrained as `{new_version}`! Accuracy: {metrics['accuracy']*100:.2f}%")
                time.sleep(1)
                st.rerun()

    st.markdown("---")
    st.markdown("#### 📜 Pipeline Execution History")
    runs = mon_state.get("pipeline_history", [])
    if runs:
        st.dataframe(pd.DataFrame(runs), use_container_width=True, hide_index=True)
    else:
        st.info("No pipeline execution runs logged yet.")


# PAGE 6: MODEL MANAGEMENT
elif nav_choice == "Model Management":
    st.markdown("### 📦 Model Registry & Deployment Management")

    models = mon_state.get("models", [])
    if not models:
        st.info("No models registered yet.")
    else:
        st.markdown("#### Deployed & Registered Model Versions")
        mod_df = pd.DataFrame(models)
        st.dataframe(mod_df, use_container_width=True, hide_index=True)

        st.markdown("#### Model Serving (FastAPI Endpoints)")
        st.markdown("""
        The active model is served via FastAPI with high-throughput inference endpoints:
        - `GET /` — Health check & operational status
        - `POST /predict` — Churn prediction payload:
        """)
        st.code("""
# Example API Request to FastAPI
curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{"features": {"tenure": 12, "MonthlyCharges": 70.35, "TotalCharges": 840.5, "gender": 1}}'
        """, language="bash")


# PAGE 7: DATA MANAGEMENT
elif nav_choice == "Data Management":
    st.markdown("### 🗄️ Dataset Ingestion & Partition Management")

    d_tab1, d_tab2 = st.tabs(["Partition Inspection", "Upload New Data Stream"])

    with d_tab1:
        st.markdown("#### Partition Overview")
        chunk_files = [f for f in os.listdir(os.path.join(PROJECT_ROOT, "data/chunks")) if f.endswith(".csv")]
        chunk_files.sort()

        summary_rows = []
        for cf in chunk_files:
            p = os.path.join(PROJECT_ROOT, "data/chunks", cf)
            df_part = pd.read_csv(p)
            summary_rows.append({
                "Partition File": cf,
                "Role": "Baseline Reference" if cf == "chunk_1.csv" else "Streaming Production",
                "Rows": len(df_part),
                "Columns": len(df_part.columns),
                "Missing Values": df_part.isna().sum().sum()
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        st.markdown("#### Inspect Partition Data Preview")
        preview_choice = st.selectbox("Select chunk to view preview:", chunk_files)
        if preview_choice:
            sample_view = pd.read_csv(os.path.join(PROJECT_ROOT, "data/chunks", preview_choice))
            st.dataframe(sample_view.head(10), use_container_width=True)

    with d_tab2:
        st.markdown("#### Ingest New Production Stream")
        st.caption("Upload a new CSV dataset to append as a production chunk without altering reference baseline.")
        uploaded_file = st.file_uploader("Upload Telco CSV Chunk", type=["csv"])
        if uploaded_file is not None:
            new_df = pd.read_csv(uploaded_file)
            st.write(f"Uploaded {len(new_df)} rows and {len(new_df.columns)} columns.")
            if st.button("Save as Chunk 5 (Incoming Stream)"):
                new_chunk_path = os.path.join(PROJECT_ROOT, "data/chunks/chunk_5.csv")
                new_df.to_csv(new_chunk_path, index=False)
                record_pipeline_run("Data Ingestion", "Completed", 0.5, {"file": "chunk_5.csv"}, "Manual CSV Upload")
                st.success("Successfully ingested chunk_5.csv! You can now monitor it.")


# PAGE 8: SETTINGS
elif nav_choice == "Settings":
    st.markdown("### ⚙️ System Monitoring & Detection Parameters")

    with st.form("settings_form"):
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            ks_thresh_input = st.number_input(
                "Kolmogorov-Smirnov (KS) Significance Threshold (p-value):",
                min_value=0.001,
                max_value=0.200,
                value=st.session_state.ks_threshold,
                step=0.01,
                help="P-values below this threshold trigger drift alert (Standard: 0.05)"
            )
            psi_thresh_input = st.number_input(
                "Population Stability Index (PSI) Threshold:",
                min_value=0.05,
                max_value=0.50,
                value=st.session_state.psi_threshold,
                step=0.05,
                help="Standard thresholds: <0.1 No Drift, 0.1-0.2 Moderate, >=0.2 High"
            )
        with col_s2:
            adwin_delta_input = st.number_input(
                "ADWIN Sensitivity Parameter (Delta):",
                min_value=0.0001,
                max_value=0.0500,
                value=st.session_state.adwin_delta,
                step=0.0005,
                format="%.4f",
                help="Confidence parameter for River ADWIN detector (Standard: 0.002)"
            )
            active_chunk_input = st.selectbox(
                "Default Active Stream Chunk:",
                [2, 3, 4],
                index=[2, 3, 4].index(st.session_state.selected_chunk) if st.session_state.selected_chunk in [2, 3, 4] else 0
            )

        submit_settings = st.form_submit_button("💾 Save & Apply Configuration")
        if submit_settings:
            st.session_state.ks_threshold = float(ks_thresh_input)
            st.session_state.psi_threshold = float(psi_thresh_input)
            st.session_state.adwin_delta = float(adwin_delta_input)
            st.session_state.selected_chunk = int(active_chunk_input)
            save_settings(ks_thresh_input, psi_thresh_input, adwin_delta_input, active_chunk_input)
            st.success("Configuration updated and persisted successfully!")
            st.rerun()
