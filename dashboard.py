"""
dashboard.py — Streamlit SOC Alert Dashboard for InsiderThreatDetector.

Run with:
    streamlit run dashboard.py -- \
        --report   shap_output/soc_alert_report.parquet \
        --detail   shap_output/soc_alert_detail.json \
        --plots    shap_output/waterfall_plots/ \
        --importance shap_output/global_feature_importance.csv

LAYOUT
──────
┌─────────────────────────────────────────────────┐
│  Sidebar: filters (date, threat category, user) │
├──────────────┬──────────────────────────────────┤
│  KPI row     │  confirmed / review / gap counts  │
├──────────────┴──────────────────────────────────┤
│  Alert table (sortable, colour-coded)           │
├─────────────────────────────────────────────────┤
│  [Click a row] → Detail panel                   │
│    • Narrative                                  │
│    • SHAP waterfall plot                        │
│    • Full feature contribution table            │
│    • MAD flag breakdown                        │
├─────────────────────────────────────────────────┤
│  Global feature importance bar chart            │
└─────────────────────────────────────────────────┘

WHY EACH DESIGN DECISION IS MADE:
    [DASH-01] st.cache_data on all file loads — Streamlit reruns the full
              script on every widget interaction. Without caching, the parquet
              and JSON reload on every filter change. On a 50k-row report that's
              a 2-3 second freeze on every click.

    [DASH-02] Parquet loaded with explicit dtype enforcement — activity_date
              forced to datetime so date filters work correctly.

    [DASH-03] Alert table uses st.data_editor, not st.dataframe — this gives
              the analyst a "select row" checkbox column which drives the
              detail panel without needing custom JS.

    [DASH-04] Detail panel only renders when exactly one row is selected —
              prevents the panel from showing stale data when the analyst
              changes filters.

    [DASH-05] SHAP waterfall plot loaded from disk path in the detail JSON —
              not regenerated in the dashboard. Regenerating SHAP in the
              dashboard would require the model to be loaded in Streamlit,
              turning a display layer into an inference layer. Wrong separation
              of concerns.

    [DASH-06] Global feature importance uses Plotly, not matplotlib — Plotly
              charts are interactive (hover, zoom) in Streamlit. Matplotlib
              static images are not.

    [DASH-07] Colour coding in the alert table: confirmed_threat = red,
              high_risk_review = orange, telemetry_gap = yellow. Applied via
              pandas Styler, not custom HTML, so it survives Streamlit version
              upgrades.

    [DASH-08] All file paths taken from CLI args, not hardcoded. This makes
              the dashboard environment-agnostic — works locally, in Docker,
              and on Streamlit Cloud without code changes.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
import os
import time


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG — must be the first Streamlit call
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title  = "Insider Threat SOC Dashboard",
    page_icon   = "🛡️",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
# Grab paths from the environment 
# ══════════════════════════════════════════════════════════════════════════════

REPORT_PATH     = os.getenv("SOC_REPORT_PATH", "shap_output3/soc_alert_report.parquet")
DETAIL_PATH     = os.getenv("SOC_DETAIL_PATH", "shap_output3/soc_alert_detail.json")
PLOTS_DIR       = os.getenv("SOC_PLOTS_DIR", "shap_output3/waterfall_plots/")
IMPORTANCE_PATH = os.getenv("SOC_IMPORTANCE_PATH", "shap_output3/global_feature_importance.csv")


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS — all cached
# ══════════════════════════════════════════════════════════════════════════════

#@st.cache_data
def load_report(path: str) -> pd.DataFrame:
    """
    [DASH-01] Cached parquet load.
    [DASH-02] Explicit dtype enforcement after load.

    activity_date: parquet stores as timestamp[ns] but some writers emit
    strings. We normalise to datetime and then to date for display clarity.
    iso_raw_score: keep as float64 — displayed to 4 decimal places.
    threat categories: keep as int (0/1) — used for filtering and colouring.
    """
    # --- TRUTH SERUM v3 ---
    #st.sidebar.error("### 🚨 FILE TRUTH SERUM")
    #abs_path = Path(path).resolve()
    #st.sidebar.write("**Absolute Path:**", str(abs_path))
    #
    #if abs_path.exists():
    #    # Foolproof timestamp formatting
    #    mod_time_str = time.ctime(os.path.getmtime(abs_path))
    #    st.sidebar.write("**Last Modified:**", mod_time_str)
    #else:
    #    st.sidebar.error("FILE DOES NOT EXIST AT THIS PATH!")
    #st.sidebar.markdown("---")
    # ----------------------

    df = pd.read_parquet(path, engine="pyarrow")
    # [FIX] Fill NaN scores so the slider doesn't drop Telemetry Gaps
    if "iso_raw_score" in df.columns:
        df["iso_raw_score"] = df["iso_raw_score"].fillna(0.0)

    # Normalise activity_date to Python date for cleaner display.
    # Must use errors='coerce' + explicit string handling because the parquet
    # may store activity_date as a raw string ("2011-01-03") rather than a
    # timestamp. pd.to_datetime handles both cases; .dt.date strips time component.
    if "activity_date" in df.columns:
        # Fill missing dates with today's date so they are never dropped
        df["activity_date"] = pd.to_datetime(
            df["activity_date"], errors="coerce"
        ).fillna(pd.Timestamp("today")).dt.date
        
    # The .notna() line has been completely removed

    # Derive boolean flag columns from threat_category string.
    # The SOC report parquet stores threat_category as a string
    # ("confirmed_threat", "high_risk_review", "review") rather than
    # separate boolean columns. The sidebar filter and colour coding
    # both depend on boolean columns — derive them here.
    if "threat_category" in df.columns:
        # Force string conversion and strip whitespace
        clean_cat = df["threat_category"].astype(str).str.strip()
        df["confirmed_threat"]   = clean_cat == "confirmed_threat"
        df["high_risk_review"]   = clean_cat == "high_risk_review"
        df["telemetry_gap_flag"] = clean_cat.isin(["review", "telemetry_gap_flag"])
    else:
        # Fallback: cast existing int columns to bool if present
        for col in ["confirmed_threat", "high_risk_review", "telemetry_gap_flag"]:
            if col in df.columns:
                df[col] = df[col].astype(bool)

    return df


@st.cache_data
def load_detail(path: str) -> dict:
    """
    [DASH-01] Cached JSON load. Returns a dict keyed by (user, activity_date)
    string tuples for O(1) lookup when the analyst selects a row.

    Why dict, not list?
    The detail panel needs to find one specific alert given a user + date.
    Searching a list of 2,000 items on every panel render is O(n).
    Dict lookup is O(1). This matters when the analyst is rapidly clicking
    through alerts.
    """
    with open(path) as f:
        raw = json.load(f)

    # Index by (user, activity_date) string key for fast lookup
    indexed = {}
    for record in raw:
        key = (str(record["user"]), str(record["activity_date"]))
        indexed[key] = record

    return indexed


@st.cache_data
def load_importance(path: str) -> pd.DataFrame:
    """[DASH-01] Cached CSV load for global feature importance."""
    return pd.read_csv(path)


# ══════════════════════════════════════════════════════════════════════════════
# STYLING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# [DASH-07] Colour mapping for threat categories
THREAT_COLOURS = {
    "confirmed_threat": "#FF4B4B",   # Streamlit red
    "high_risk_review": "#FFA500",   # Orange
    "telemetry_gap_flag":    "#FFD700",   # Yellow       # changed from "telemetry_gap"
}

def _style_threat_row_display(row: pd.Series) -> list[str]:
    severity = str(row.get("severity", ""))
    if "CONFIRMED" in severity:
        colour = "background-color: #3d0000; color: #FF4B4B"
    elif "REVIEW" in severity:
        colour = "background-color: #3d2000; color: #FFA500"
    elif "DATA GAP" in severity:
        colour = "background-color: #3d3000; color: #FFD700"
    else:
        colour = ""
    return [colour] * len(row)


def _threat_badge(row: pd.Series) -> str:
    """Returns a short label string for the threat category column."""
    if row.get("confirmed_threat", False):
        return "🔴 CONFIRMED"
    elif row.get("high_risk_review", False):
        return "🟠 REVIEW"
    elif row.get("telemetry_gap_flag", False):
        return "🟡 DATA GAP"
    return "⚪ CLEAN"


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — FILTERS
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renders sidebar filters and returns the filtered DataFrame.

    Filter order matters: each filter reduces the dataset before the next
    filter's options are computed. This prevents the "no results" state where
    a user selects a date then a user who has no alerts on that date.
    """
    st.sidebar.header("🔍 Filters")

    # ── Date range ────────────────────────────────────────────────────────────
    # Uses string comparison against activity_date directly — avoids the
    # datetime.date vs string type mismatch that silently excludes all rows
    # when the parquet stores activity_date as a string.
    if "activity_date" in df.columns:
        df["activity_date_str"] = df["activity_date"].astype(str)
        all_dates = sorted(df["activity_date_str"].unique())

        if len(all_dates) >= 2:
            col1, col2 = st.sidebar.columns(2)
            start_date = col1.selectbox("From", all_dates, index=0)
            end_date   = col2.selectbox("To",   all_dates, index=len(all_dates)-1)

            df = df[
                (df["activity_date_str"] >= start_date) &
                (df["activity_date_str"] <= end_date)
            ]

        df = df.drop(columns=["activity_date_str"])

    # ── Threat category ───────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    show_confirmed = st.sidebar.checkbox("🔴 Confirmed Threats", value=True)
    show_review    = st.sidebar.checkbox("🟠 High Risk Review",  value=True)
    show_gap       = st.sidebar.checkbox("🟡 Telemetry Gaps",    value=True)

    category_mask = pd.Series(False, index=df.index)
    if show_confirmed and "confirmed_threat"  in df.columns:
        category_mask |= df["confirmed_threat"]
    if show_review    and "high_risk_review"  in df.columns:
        category_mask |= df["high_risk_review"]
    if show_gap       and "telemetry_gap_flag" in df.columns:
        category_mask |= df["telemetry_gap_flag"]
    df = df[category_mask]

    # ── User search ───────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    user_search = st.sidebar.text_input("Search by User ID", value="")
    if user_search.strip() and "user" in df.columns:
        df = df[df["user"].astype(str).str.contains(user_search.strip(), case=False)]

    # ── Anomaly score threshold ───────────────────────────────────────────────
    st.sidebar.markdown("---")
    if "iso_raw_score" in df.columns and len(df) > 0:
        score_min = float(df["iso_raw_score"].min())
        score_max = float(df["iso_raw_score"].max())

        if score_min < score_max:
            score_threshold = st.sidebar.slider(
                "Max ISO Forest Score (lower = more anomalous)",
                min_value = score_min,
                max_value = score_max,
                value     = score_max,
                step      = round((score_max - score_min) / 100, 4),
            )
            df = df[df["iso_raw_score"] <= score_threshold]

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Showing **{len(df)}** alerts after filters.")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════════════

def render_kpis(df_filtered: pd.DataFrame, df_full: pd.DataFrame) -> None:
    """
    Renders the top KPI metrics row.
    Shows filtered counts alongside full-dataset totals for context.
    """
    n_confirmed = int(df_filtered.get("confirmed_threat",  pd.Series(0)).sum()) if "confirmed_threat"  in df_filtered.columns else 0
    n_review    = int(df_filtered.get("high_risk_review",  pd.Series(0)).sum()) if "high_risk_review"  in df_filtered.columns else 0
    n_gap       = int(df_filtered.get("telemetry_gap_flag",pd.Series(0)).sum()) if "telemetry_gap_flag" in df_filtered.columns else 0
    n_total     = len(df_filtered)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        label = "🔴 Confirmed Threats",
        value = n_confirmed,
        delta = f"of {int(df_full['confirmed_threat'].sum()) if 'confirmed_threat' in df_full.columns else '?'} total",
        delta_color = "inverse",
    )
    col2.metric(
        label = "🟠 High Risk Reviews",
        value = n_review,
        delta = f"of {int(df_full['high_risk_review'].sum()) if 'high_risk_review' in df_full.columns else '?'} total",
        delta_color = "inverse",
    )
    col3.metric(
        label = "🟡 Telemetry Gaps",
        value = n_gap,
        delta = f"of {int(df_full['telemetry_gap_flag'].sum()) if 'telemetry_gap_flag' in df_full.columns else '?'} total",
        delta_color = "inverse",
    )
    col4.metric(
        label = "Total Alerts (filtered)",
        value = n_total,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ALERT TABLE
# ══════════════════════════════════════════════════════════════════════════════

def render_alert_table(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Renders the main alert table and returns the selected row, if any.

    [DASH-03] Uses st.dataframe with on_select="rerun" (Streamlit ≥ 1.35).
    This is the correct approach for row selection — avoids the anti-pattern
    of putting a selectbox next to the table, which forces the analyst to
    cross-reference two separate UI elements.

    Returns the selected row as a single-row DataFrame, or None.
    """
    st.subheader("🚨 Alert Table")

    if df.empty:
        st.info("No alerts match the current filters.")
        return None

    # Build the display version — add badge column, drop raw flag columns
    display_cols = ["user", "activity_date", "iso_raw_score", "mad_flag_ratio"]
    display_cols += [c for c in df.columns if c.startswith("top1_") or c.startswith("top2_")]
    display_cols  = [c for c in display_cols if c in df.columns]

    df_display = df[display_cols].copy()
    df_display.insert(0, "severity", df.apply(_threat_badge, axis=1))
    df_display["iso_raw_score"] = df_display["iso_raw_score"].round(4)
    df_display["mad_flag_ratio"]        = df_display["mad_flag_ratio"].round(3)

    # [DASH-03] Row selection via st.dataframe
    event = st.dataframe(
        df_display.style.apply(_style_threat_row_display, axis=1),
        use_container_width = True,
        hide_index          = True,
        on_select           = "rerun",
        selection_mode      = "single-row",
        column_config = {
            "severity": st.column_config.TextColumn("Severity", width="small"),
            "iso_raw_score": st.column_config.NumberColumn(
                "ISO Score", format="%.4f", help="Lower = more anomalous"
            ),
            "mad_flag_ratio": st.column_config.NumberColumn(
                "MAD Ratio", format="%.3f"
            ),
        }
    )

    selected_rows = event.selection.rows if hasattr(event, "selection") else []
    if selected_rows:
        return df.iloc[selected_rows]

    return None


# ══════════════════════════════════════════════════════════════════════════════
# DETAIL PANEL
# ══════════════════════════════════════════════════════════════════════════════

def render_detail_panel(
    selected_row: pd.DataFrame,
    detail_index: dict,
    plots_dir: str,
) -> None:
    """
    Renders the full alert detail panel for the selected row.

    [DASH-04] Only renders when exactly one row is selected.
    [DASH-05] Loads the waterfall plot from disk — does not regenerate SHAP.

    Panels:
        1. Narrative (plain-English summary)
        2. SHAP waterfall plot
        3. Full feature contribution table (all features, not just top-N)
        4. MAD flag breakdown
    """
    st.markdown("---")
    st.subheader("🔬 Alert Detail")

    row      = selected_row.iloc[0]
    user     = str(row["user"])
    act_date = str(row["activity_date"])
    key      = (user, act_date)

    detail = detail_index.get(key)

    if detail is None:
        st.warning(
            f"Detail record not found for ({user}, {act_date}). "
            f"The detail JSON may be from a different inference run than the report."
        )
        return

    # ── Panel header ──────────────────────────────────────────────────────────
    threat_cat = detail.get("threat_category", "unknown")
    colour     = THREAT_COLOURS.get(threat_cat, "#FFFFFF")

    st.markdown(
        f"<div style='border-left: 4px solid {colour}; padding-left: 12px;'>"
        f"<h4 style='color:{colour}'>{threat_cat.upper().replace('_',' ')}</h4>"
        f"<p><b>User:</b> {user} &nbsp;|&nbsp; <b>Date:</b> {act_date} &nbsp;|&nbsp; "
        f"<b>Baseline:</b> {detail.get('baseline_source','?')}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Narrative ─────────────────────────────────────────────────────────────
    st.markdown("#### 📋 Analyst Narrative")
    st.info(detail.get("narrative", "No narrative available."))

    # ── Two-column layout: waterfall + MAD breakdown ──────────────────────────
    left_col, right_col = st.columns([1.6, 1])

    with left_col:
        st.markdown("#### 📊 SHAP Waterfall Plot")

        # [DASH-05] Load from disk — the path was stored in the detail JSON
        # by _generate_waterfall_plot() in shap_explainer.py
        plot_path = detail.get("plot_path")
        if plot_path and Path(plot_path).exists():
            img = Image.open(plot_path)
            st.image(img, use_container_width=True)
        else:
            st.caption(
                "Waterfall plot not found. "
                "Re-run with generate_plots=True or check the plots directory."
            )

    with right_col:
        st.markdown("#### 🚩 MAD Flag Breakdown")

        # Extract per-feature MAD flags from the main report row
        flag_cols = {
            col.replace("_flag", ""): bool(val)
            for col, val in row.items()
            if col.endswith("_flag") and col not in
               {"confirmed_threat", "iso_forest_flag", "mad_critical_flag",
                "data_quality_risk", "telemetry_gap_flag", "high_risk_review"}
        }

        if flag_cols:
            flagged_features   = [f for f, v in flag_cols.items() if v]
            unflagged_features = [f for f, v in flag_cols.items() if not v]

            if flagged_features:
                st.markdown("**Flagged by MAD model:**")
                for feat in flagged_features:
                    st.markdown(f"⚠️ `{feat}`")
            else:
                st.markdown("*No MAD flags on this record.*")

            with st.expander("Unflagged features"):
                for feat in unflagged_features:
                    st.markdown(f"✅ `{feat}`")
        else:
            st.caption("No per-feature flag columns found in report.")

    # ── Full feature contribution table ───────────────────────────────────────
    st.markdown("#### 📐 Full Feature Contributions (all features, ranked by |SHAP|)")

    all_contribs = detail.get("all_contributions", [])
    if all_contribs:
        contrib_df = pd.DataFrame([
            {
                "Feature":      c["feature_name"],
                "Raw Value":    round(c["raw_value"], 3),
                "Baseline":     round(c["user_baseline"], 3),
                "Z-Score":      round(c["self_z_score"], 2),
                "SHAP Value":   round(c["shap_value"], 5),
                "Direction":    c["direction"],
                "MAD Flagged":  "⚠️ Yes" if c["mad_flagged"] else "✅ No",
                "Explanation":  c["plain_english"],
            }
            for c in all_contribs
        ])

        # Colour Z-Score column: high z = red, normal = green
        def _colour_zscore(val):
            if val > 3.5:
                return "color: #FF4B4B; font-weight: bold"
            elif val > 2.0:
                return "color: #FFA500"
            return "color: #00C853"

        styled = (
            contrib_df.style
            .map(_colour_zscore, subset=["Z-Score"])
            .format({"SHAP Value": "{:.5f}", "Z-Score": "{:.2f}"})
        )

        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Explanation": st.column_config.TextColumn(width="large"),
            }
        )
    else:
        st.caption("No contribution data available.")


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

def render_global_importance(importance_df: pd.DataFrame) -> None:
    """
    [DASH-06] Renders the global SHAP feature importance as an interactive
    Plotly bar chart.

    WHY Plotly, not matplotlib:
    matplotlib produces a static image in Streamlit. Plotly produces an
    interactive chart where the analyst can hover to see exact values, zoom
    in on closely-ranked features, and screenshot specific views. For a
    feature importance chart used in model auditing and stakeholder reporting,
    interactivity is essential.
    """
    st.markdown("---")
    st.subheader("🌐 Global Feature Importance (SHAP)")
    st.caption(
        "Mean absolute SHAP value across the training population. "
        "Higher = this feature drives anomaly scores more consistently across all users."
    )

    if importance_df.empty:
        st.info("No global importance data available.")
        return

    fig = px.bar(
        importance_df.sort_values("mean_abs_shap", ascending=True),
        x           = "mean_abs_shap",
        y           = "feature",
        orientation = "h",
        color       = "mean_abs_shap",
        color_continuous_scale = "Reds",
        labels      = {
            "mean_abs_shap": "Mean |SHAP Value|",
            "feature":       "Feature",
        },
        title       = "Global SHAP Feature Importance",
    )

    fig.update_layout(
        height          = max(400, len(importance_df) * 30),
        coloraxis_showscale = False,
        plot_bgcolor    = "rgba(0,0,0,0)",
        paper_bgcolor   = "rgba(0,0,0,0)",
        font            = dict(color="#FAFAFA"),
        xaxis           = dict(gridcolor="#333333"),
        yaxis           = dict(gridcolor="#333333"),
    )

    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TIMELINE CHART — alerts over time
# ══════════════════════════════════════════════════════════════════════════════

def render_timeline(df: pd.DataFrame) -> None:
    """
    Renders a stacked bar chart of alert counts by date and severity.

    WHY this chart:
    The timeline is the first thing a SOC manager looks at in a morning briefing.
    "Did alerts spike overnight?" is the first question. A flat bar chart by
    user answers "who?" — the timeline answers "when?" Both are required.
    """
    if "activity_date" not in df.columns or df.empty:
        return

    st.markdown("---")
    st.subheader("📅 Alert Timeline")

    timeline_rows = []
    for _, row in df.iterrows():
        date = row["activity_date"]
        if row.get("confirmed_threat", False):
            timeline_rows.append({"date": date, "category": "Confirmed Threat"})
        elif row.get("high_risk_review", False):
            timeline_rows.append({"date": date, "category": "High Risk Review"})
        elif row.get("telemetry_gap_flag", False):
            timeline_rows.append({"date": date, "category": "Telemetry Gap"})

    if not timeline_rows:
        return

    timeline_df = (
        pd.DataFrame(timeline_rows)
        .groupby(["date", "category"])
        .size()
        .reset_index(name="count")
    )

    fig = px.bar(
        timeline_df,
        x       = "date",
        y       = "count",
        color   = "category",
        color_discrete_map = {
            "Confirmed Threat": "#FF4B4B",
            "High Risk Review": "#FFA500",
            "Telemetry Gap":    "#FFD700",
        },
        barmode = "stack",
        labels  = {"date": "Activity Date", "count": "Alert Count"},
        title   = "Alerts by Date and Category",
    )

    fig.update_layout(
        plot_bgcolor  = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        font          = dict(color="#FAFAFA"),
        xaxis         = dict(gridcolor="#333333"),
        yaxis         = dict(gridcolor="#333333"),
    )

    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.title("🛡️ Insider Threat SOC Dashboard")
    st.caption(
        "Powered by InsiderThreatDetector (Self-as-Baseline MAD + Isolation Forest) "
        "with SHAP explainability."
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    try:
    
        df_full       = load_report(REPORT_PATH)
        detail_index  = load_detail(DETAIL_PATH)
        importance_df = load_importance(IMPORTANCE_PATH)

    except FileNotFoundError as e:
        st.error(
            f"Required data file not found: {e}\n\n"
            f"Check your environment variables or run `run_shap_inference()` first."
        )
        st.stop()

    # ── Sidebar filters ───────────────────────────────────────────────────────
    df_filtered = render_sidebar(df_full)

    # ── KPI row ───────────────────────────────────────────────────────────────
    render_kpis(df_filtered, df_full)

    # ── Timeline ──────────────────────────────────────────────────────────────
    render_timeline(df_filtered)

    # ── Alert table + detail panel ────────────────────────────────────────────
    selected = render_alert_table(df_filtered)

    # [DASH-04] Only show detail panel when exactly one row is selected
    if selected is not None and len(selected) == 1:
        render_detail_panel(
            selected_row = selected,
            detail_index = detail_index,
            plots_dir    = PLOTS_DIR,
        )
    elif selected is not None and len(selected) > 1:
        st.info("Select a single row to view its SHAP detail.")

    # ── Global feature importance ─────────────────────────────────────────────
    render_global_importance(importance_df)


if __name__ == "__main__":
    main()