"""
SOC Alert Triage & Investigation Dashboard
-------------------------------------------
Reads scored_features_latest.parquet produced by inference.py.

Fixes applied vs. original:
  — SHAP columns are discovered dynamically (no hardcoded feature names).
       Any feature the model was trained on gets its own bar in the SHAP chart.
  — KPIs now surface dual-model signals: confirmed_threat, high_risk_review,
       data_loss_ioc — not a batch-relative risk_score threshold.
  — Every section guards against empty DataFrames.  "No threats today"
       is displayed as a success state, not a Python crash.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Insider Threat SOC", layout="wide")


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_scored_data(file_path: str = "features/scored_features_latest.parquet") -> pd.DataFrame:
    """
    Loads the feature + score matrix produced by inference.py.

    Handles:
    - MultiIndex flattening (user, activity_date may be in index)
    - Timestamp → date conversion for Plotly
    - Missing file → informative error, not a crash
    """
    try:
        df = pd.read_parquet(file_path)
    except FileNotFoundError:
        st.error(f"🚨 Data file not found: `{file_path}`")
        st.warning(
            "Run the inference pipeline first:\n"
            "```\npython inference.py\n```"
        )
        st.stop()

    # Flatten MultiIndex if user/activity_date were saved as index
    if df.index.names != [None]:
        df = df.reset_index()

    # Normalise activity_date to pure date objects
    if "activity_date" in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df["activity_date"]):
            df["activity_date"] = df["activity_date"].dt.date

    return df.sort_values("activity_date")


# ─────────────────────────────────────────────────────────────────────────────
# SHAP COLUMN DISCOVERY
# fix: we never hardcode feature names. The dashboard discovers every
# `shap_*` column that inference.py wrote and renders them all.
# ─────────────────────────────────────────────────────────────────────────────
def get_shap_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.startswith("shap_")]


def shap_display_name(col: str) -> str:
    """Converts 'shap_off_hours_ratio' → 'Off Hours Ratio'."""
    return col.replace("shap_", "").replace("_", " ").title()


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
df = load_scored_data()
today = df["activity_date"].max()
latest_data = df[df["activity_date"] == today].copy()

shap_cols = get_shap_columns(df)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("🚨 SOC Alert Triage & Investigation")
st.caption(f"Data snapshot: **{today}** | Total user-days in window: **{len(df):,}**")
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — RANKED TRIAGE QUEUE
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("1. Active Threat Queue (Action Required)")
st.write("Users ranked by dual-model threat signal and telemetry integrity.")

# Sort: confirmed threats first, then high_risk_review, then by risk_score
triage_queue = latest_data.sort_values(
    by=["confirmed_threat", "high_risk_review", "risk_score"],
    ascending=[False, False, False],
)

# ── KPI row ──────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

confirmed_count  = int(triage_queue["confirmed_threat"].sum())
review_count     = int(triage_queue["high_risk_review"].sum())
ioc_count        = int(triage_queue["data_loss_ioc"].sum())
high_score_users = int((triage_queue["risk_score"] > 0.75).sum())

col1.metric("Users Active Today",         len(triage_queue))
col2.metric("✅ Confirmed Threats",        confirmed_count,
            delta=confirmed_count, delta_color="inverse")
col3.metric("⚠️ High Risk Review",         review_count,
            delta=review_count,    delta_color="inverse")
col4.metric("🔇 Data Loss IOCs",           ioc_count,
            delta=ioc_count,       delta_color="inverse")
col5.metric("risk_score > 0.75",           high_score_users,
            delta=high_score_users, delta_color="inverse")

# ── Triage table ─────────────────────────────────────────────────────────────
# fix: guard against an empty triage queue (the normal-day case)
if triage_queue.empty:
    st.success("✅ No anomalies detected today. Fleet is healthy.")
else:
    display_cols = [
        c for c in [
            "user", "risk_score", "confirmed_threat",
            "high_risk_review", "data_loss_ioc", "data_quality_risk",
            "mad_critical_flag", "iso_forest_flag", "total_events"
        ]
        if c in triage_queue.columns
    ]
    st.dataframe(
        triage_queue[display_cols]
        .style.background_gradient(subset=["risk_score"], cmap="Reds"),
        use_container_width=True,
        hide_index=True,
    )

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — USER INVESTIGATION DEEP DIVE
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("2. User Investigation Deep Dive")
st.write(
    "Select a user from the triage queue to analyse their behavioural "
    "trajectory and ML feature attributions."
)

# fix: guard against empty queue before iloc[0]
if triage_queue.empty:
    st.info("No users to investigate today.")
else:
    selected_user = st.selectbox(
        "Investigating User:",
        options=triage_queue["user"].unique(),
        index=0,
    )

    user_history = df[df["user"] == selected_user]

    # fix: guard against the selected user having no row for today
    user_today_rows = user_history[user_history["activity_date"] == today]
    if user_today_rows.empty:
        st.warning(f"No data for user `{selected_user}` on {today}.")
    else:
        user_today = user_today_rows.iloc[0]

        chart_col1, chart_col2 = st.columns(2)

        # ── SHAP attribution chart ────────────────────────────────────────
        with chart_col1:
            st.markdown("**Why did the model flag this? (SHAP Analysis)**")

            # fix: build chart from discovered shap_* columns, not hardcoded names
            if not shap_cols:
                st.info("No SHAP columns found in data. Re-run inference.py.")
            else:
                shap_data = pd.DataFrame({
                    "Feature":      [shap_display_name(c) for c in shap_cols],
                    "Contribution": [user_today.get(c, 0.0) for c in shap_cols],
                })
                shap_data["Color"] = np.where(
                    shap_data["Contribution"] > 0, "Crimson", "LightGray"
                )
                # Sort by absolute contribution so biggest drivers are at top
                shap_data = shap_data.reindex(
                    shap_data["Contribution"].abs().sort_values(ascending=True).index
                )

                fig_shap = px.bar(
                    shap_data,
                    x="Contribution", y="Feature",
                    orientation="h",
                    title=f"Feature Impact on Risk Score ({today})",
                    color="Color",
                    color_discrete_map="identity",
                )
                fig_shap.update_layout(
                    xaxis_title="SHAP Contribution (positive = increases risk)",
                    yaxis_title="",
                    showlegend=False,
                )
                st.plotly_chart(fig_shap, use_container_width=True)

        # ── 14-day risk trajectory ────────────────────────────────────────
        with chart_col2:
            st.markdown("**How is this unfolding? (14-Day Trajectory)**")

            fig_line = px.line(
                user_history,
                x="activity_date", y="risk_score",
                markers=True,
                title=f"Risk Trend: {selected_user}",
            )
            # 0.75 threshold is now meaningful (calibrated percentile)
            fig_line.add_hline(
                y=0.75, line_dash="dash", line_color="red",
                annotation_text="75th-percentile threshold",
            )
            fig_line.update_layout(
                yaxis_range=[0, 1.05],
                xaxis_title="",
                yaxis_title="Calibrated Risk Score (training percentile)",
            )
            st.plotly_chart(fig_line, use_container_width=True)

        # ── Dual-model signal breakdown ───────────────────────────────────
        st.markdown("**Model Signal Breakdown**")
        signal_cols = [
            c for c in [
                "mad_score_count", "mad_critical_flag",
                "iso_forest_raw_score", "iso_forest_flag",
                "confirmed_threat", "high_risk_review", "data_loss_ioc",
            ]
            if c in user_today.index
        ]
        st.table(
            pd.DataFrame(
                user_today[signal_cols].rename(index={
                    "mad_score_count":       "MAD: features flagged",
                    "mad_critical_flag":     "MAD: critical (>=2 features)",
                    "iso_forest_raw_score":  "ISO Forest: raw score",
                    "iso_forest_flag":       "ISO Forest: flagged",
                    "confirmed_threat":      "Confirmed Threat (both models, clean data)",
                    "high_risk_review":      "High Risk Review (both models, dirty data)",
                    "data_loss_ioc":         "Data Loss IOC (telemetry gap only)",
                })
            ).rename(columns={0: "Value"})
        )

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — FLEET-WIDE HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("3. Systemic Threat Analysis (Blast Radius)")
st.write(
    "Scan for coordinated attacks, lateral movement campaigns, "
    "or systemic IT misconfigurations."
)

if df.empty:
    st.info("No data available for heatmap.")
else:
    fig_heat = px.density_heatmap(
        df,
        x="activity_date", y="user", z="risk_score",
        histfunc="max",
        color_continuous_scale="Reds",
        title="Fleet Anomaly Heatmap (calibrated risk_score)",
    )
    fig_heat.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_colorbar_title="Max Risk",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Confirmed threat timeline across all users
    if "confirmed_threat" in df.columns and df["confirmed_threat"].sum() > 0:
        st.markdown("**Confirmed Threat Events Over Time**")
        threat_timeline = (
            df.groupby("activity_date")["confirmed_threat"]
            .sum()
            .reset_index()
        )
        fig_timeline = px.bar(
            threat_timeline,
            x="activity_date", y="confirmed_threat",
            title="Confirmed Threats per Day (both models + clean telemetry)",
            color_discrete_sequence=["crimson"],
        )
        fig_timeline.update_layout(xaxis_title="", yaxis_title="Count")
        st.plotly_chart(fig_timeline, use_container_width=True)
