"""
diagnostics.py — Model Performance & Threshold Diagnostic Suite.

Run this BEFORE concluding your model is broken. Most "missing class" issues
are threshold calibration problems, not model failures.

Usage (notebook):
    from diagnostics import run_full_diagnostic
    run_full_diagnostic(detector, df_scores, df_live_features)

Usage (script):
    python diagnostics.py \
        --model  iso_pipeline_v20250517_143022.pkl \
        --live   data/live.parquet \
        --scores shap_output/soc_alert_report.parquet   # optional, recomputes if missing
"""

import argparse
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("Diagnostics")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — RAW SCORE DISTRIBUTIONS
# Answers: are the intermediate signals even firing before fusion?
# ══════════════════════════════════════════════════════════════════════════════

def diag_score_distributions(df_scores: pd.DataFrame) -> None:
    """
    Prints distribution stats for every intermediate score column.

    If high_risk_review=0 and telemetry_gap_flag=0, this section tells you
    exactly which upstream signal is dead — data_quality_risk, mad_critical_flag,
    or iso_forest_flag — so you know which threshold to adjust.
    """
    print("\n" + "═"*65)
    print("SECTION 1 — RAW SCORE DISTRIBUTIONS")
    print("═"*65)

    # ── 1a. Flag column counts ─────────────────────────────────────────────
    flag_cols = [
        "data_quality_risk",
        "mad_critical_flag",
        "iso_forest_flag",
        "confirmed_threat",
        "high_risk_review",
        "telemetry_gap_flag",
    ]

    print("\n[1a] Flag column value counts (how many rows hit each flag):\n")
    total = len(df_scores)
    for col in flag_cols:
        if col not in df_scores.columns:
            print(f"  {col:<28} — COLUMN MISSING from df_scores")
            continue
        n    = int(df_scores[col].sum())
        pct  = 100 * n / total if total > 0 else 0
        bar  = "█" * min(int(pct * 2), 50)
        print(f"  {col:<28} {n:>7,} / {total:,}  ({pct:5.2f}%)  {bar}")

    # ── 1b. Fusion breakdown — why confirmed_threat is low ─────────────────
    print("\n[1b] Fusion logic breakdown:")
    print("     confirmed_threat  = mad_critical_flag AND iso_forest_flag AND NOT data_quality_risk")
    print("     high_risk_review  = mad_critical_flag AND iso_forest_flag AND data_quality_risk")
    print("     telemetry_gap     = data_quality_risk AND NOT confirmed AND NOT review\n")

    if all(c in df_scores.columns for c in ["mad_critical_flag", "iso_forest_flag", "data_quality_risk"]):
        both    = (df_scores["mad_critical_flag"] == 1) & (df_scores["iso_forest_flag"] == 1)
        clean   = df_scores["data_quality_risk"] == 0
        dirty   = df_scores["data_quality_risk"] == 1

        print(f"  Both models agree (mad AND iso):      {both.sum():>7,}")
        print(f"    → clean telemetry (→ confirmed):    {(both & clean).sum():>7,}")
        print(f"    → dirty telemetry (→ high_risk):    {(both & dirty).sum():>7,}")
        print(f"  data_quality_risk alone (→ gap flag): {(dirty & ~both).sum():>7,}")

        if both.sum() == 0:
            print("\n  ⚠ DIAGNOSIS: Neither model is agreeing on ANY row.")
            print("    Either mad_critical_flag or iso_forest_flag (or both) is always 0.")
            print("    See Section 2 and Section 3 for threshold analysis.")
        elif (both & dirty).sum() == 0:
            print("\n  ⚠ DIAGNOSIS: Both models agree but telemetry is always clean.")
            print("    high_risk_review requires data_quality_risk=1.")
            print("    Your live data has very low missingness — see Section 4.")

    # ── 1c. Continuous score distributions ────────────────────────────────
    print("\n[1c] Continuous score distributions:\n")

    for col, label in [
        ("iso_forest_raw_score", "ISO Forest raw score (lower = more anomalous)"),
        ("mad_flag_ratio",       "MAD flag ratio (higher = more features flagged)"),
        ("mad_score_count",      "MAD flagged feature count per row"),
    ]:
        if col not in df_scores.columns:
            continue
        s = df_scores[col].dropna()
        print(f"  {label}")
        print(f"    min={s.min():.4f}  p1={s.quantile(.01):.4f}  p5={s.quantile(.05):.4f}  "
              f"median={s.median():.4f}  p95={s.quantile(.95):.4f}  "
              f"p99={s.quantile(.99):.4f}  max={s.max():.4f}\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MAD THRESHOLD ANALYSIS
# Answers: is critical_flag_ratio set too high?
# ══════════════════════════════════════════════════════════════════════════════

def diag_mad_threshold(detector, df_scores: pd.DataFrame) -> None:
    """
    Shows what happens to confirmed_threat count at different critical_flag_ratio
    values. If lowering the ratio dramatically increases alerts, the threshold
    is the problem — not the model.
    """
    print("\n" + "═"*65)
    print("SECTION 2 — MAD THRESHOLD ANALYSIS")
    print("═"*65)

    current_ratio = getattr(detector, "critical_flag_ratio", None)
    print(f"\n  Locked critical_flag_ratio = {current_ratio}")
    print(f"  Locked mad_threshold       = {detector.mad_threshold}")

    if "mad_flag_ratio" not in df_scores.columns:
        print("  mad_flag_ratio column missing — skipping.")
        return

    iso_flag = df_scores.get("iso_forest_flag", pd.Series(0, index=df_scores.index))
    dq_risk  = df_scores.get("data_quality_risk", pd.Series(0, index=df_scores.index))

    print("\n  Simulated confirmed_threat count at different critical_flag_ratio values:\n")
    print(f"  {'ratio':>8}  {'confirmed':>10}  {'high_risk':>10}  {'% of total':>10}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")

    for ratio in [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, current_ratio]:
        if ratio is None:
            continue
        ratio = round(ratio, 4)
        sim_mad   = (df_scores["mad_flag_ratio"] >= ratio).astype(int)
        sim_conf  = ((sim_mad == 1) & (iso_flag == 1) & (dq_risk == 0)).sum()
        sim_rev   = ((sim_mad == 1) & (iso_flag == 1) & (dq_risk == 1)).sum()
        pct       = 100 * (sim_conf + sim_rev) / len(df_scores)
        marker    = "  ← CURRENT" if ratio == current_ratio else ""
        print(f"  {ratio:>8.4f}  {sim_conf:>10,}  {sim_rev:>10,}  {pct:>9.2f}%{marker}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ISO FOREST THRESHOLD ANALYSIS
# Answers: is iso_threshold cutting off too few anomalies?
# ══════════════════════════════════════════════════════════════════════════════

def diag_iso_threshold(detector, df_scores: pd.DataFrame) -> None:
    """
    Shows what happens to flag counts at different iso_threshold values.
    The iso_threshold was set at the bottom iso_flag_percentile (default 1%)
    of training scores. If your live data distribution shifted, the threshold
    may now be too tight or too loose.
    """
    print("\n" + "═"*65)
    print("SECTION 3 — ISO FOREST THRESHOLD ANALYSIS")
    print("═"*65)

    current_thresh = getattr(detector, "iso_threshold", None)
    print(f"\n  Locked iso_threshold = {current_thresh}")
    print(f"  iso_flag_percentile  = {detector.iso_flag_percentile}%  (bottom N% of training scores)")

    if "iso_forest_raw_score" not in df_scores.columns:
        print("  iso_forest_raw_score column missing — skipping.")
        return

    raw  = df_scores["iso_forest_raw_score"]
    mad  = df_scores.get("mad_critical_flag", pd.Series(0, index=df_scores.index))
    dq   = df_scores.get("data_quality_risk",  pd.Series(0, index=df_scores.index))

    print(f"\n  Live score distribution vs training threshold:")
    print(f"    Live score min:    {raw.min():.6f}")
    print(f"    Live score median: {raw.median():.6f}")
    print(f"    Live score max:    {raw.max():.6f}")
    print(f"    Rows BELOW threshold (flagged): {(raw < current_thresh).sum():,}")
    print(f"    Rows ABOVE threshold (clean):   {(raw >= current_thresh).sum():,}")

    print(f"\n  Simulated confirmed_threat at different iso_threshold values:\n")
    print(f"  {'threshold':>10}  {'iso_flagged':>12}  {'confirmed':>10}  {'high_risk':>10}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*10}")

    # Test thresholds from the actual live score distribution percentiles
    test_percentiles = [1, 2, 5, 10, 20, 50]
    test_thresholds  = [float(np.percentile(raw, p)) for p in test_percentiles]
    if current_thresh is not None:
        test_thresholds.append(current_thresh)
    test_thresholds = sorted(set(round(t, 6) for t in test_thresholds))

    for thresh in test_thresholds:
        sim_iso  = (raw < thresh).astype(int)
        sim_conf = ((mad == 1) & (sim_iso == 1) & (dq == 0)).sum()
        sim_rev  = ((mad == 1) & (sim_iso == 1) & (dq == 1)).sum()
        marker   = "  ← CURRENT" if current_thresh and abs(thresh - current_thresh) < 1e-5 else ""
        print(f"  {thresh:>10.6f}  {sim_iso.sum():>12,}  {sim_conf:>10,}  {sim_rev:>10,}{marker}")

    # Drift check: compare live score distribution to training percentiles
    if hasattr(detector, "train_score_percentiles") and detector.train_score_percentiles:
        train_p50 = detector.train_score_percentiles[500]   # index 500 = 50th percentile
        live_p50  = float(raw.median())
        shift     = live_p50 - train_p50
        print(f"\n  Distribution drift check:")
        print(f"    Training median score: {train_p50:.6f}")
        print(f"    Live median score:     {live_p50:.6f}")
        print(f"    Shift:                 {shift:+.6f}", end="")
        if abs(shift) > 0.05:
            print(f"  ⚠ Significant drift detected — iso_threshold may need recalibration.")
        else:
            print(f"  ✓ Within acceptable range.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DATA QUALITY RISK ANALYSIS
# Answers: why is high_risk_review = 0?
# ══════════════════════════════════════════════════════════════════════════════

def diag_data_quality(detector, df_scores: pd.DataFrame, df_live: pd.DataFrame) -> None:
    """
    Diagnoses why data_quality_risk never fires (which kills high_risk_review
    and telemetry_gap_flag both).

    high_risk_review requires data_quality_risk=1. If your live data has very
    low missingness, data_quality_risk is always 0 and high_risk_review can
    never appear regardless of the model scores.
    """
    print("\n" + "═"*65)
    print("SECTION 4 — DATA QUALITY RISK ANALYSIS")
    print("═"*65)

    tol = getattr(detector, "missing_data_tolerance", 0.18)
    print(f"\n  missing_data_tolerance = {tol}  ({tol*100:.0f}% of features must be NaN to trigger)")

    if "data_quality_risk" not in df_scores.columns:
        print("  data_quality_risk column missing — skipping.")
        return

    dq_count = int(df_scores["data_quality_risk"].sum())
    print(f"  Rows with data_quality_risk=1: {dq_count:,} / {len(df_scores):,}")

    # Compute actual missing % per row from live features
    numeric_cols = getattr(detector, "_feature_schema", [])
    if len(numeric_cols) == 0:
        print("  No _feature_schema on detector — skipping per-row missingness.")
        return

    available_cols = [c for c in numeric_cols if c in df_live.columns]
    if not available_cols:
        print("  Live feature columns not found — skipping per-row missingness.")
        return

    missing_pct = df_live[available_cols].isnull().mean(axis=1)

    print(f"\n  Per-row missingness distribution (across {len(available_cols)} features):")
    print(f"    min={missing_pct.min():.4f}  median={missing_pct.median():.4f}  "
          f"mean={missing_pct.mean():.4f}  max={missing_pct.max():.4f}")
    print(f"    Rows with ANY missing value:          {(missing_pct > 0).sum():,}")
    print(f"    Rows exceeding tolerance ({tol:.0%}):    {(missing_pct > tol).sum():,}")

    if dq_count == 0:
        print(f"\n  ⚠ DIAGNOSIS: data_quality_risk never fires.")
        if missing_pct.max() < tol:
            print(f"    Your live data has very low missingness (max={missing_pct.max():.4f}).")
            print(f"    missing_data_tolerance={tol} is never breached.")
            print(f"    → high_risk_review and telemetry_gap_flag CANNOT appear.")
            print(f"    → This is expected behaviour if your live pipeline fills NaNs upstream.")
            print(f"    → If you want these classes: lower missing_data_tolerance or")
            print(f"      don't forward-fill NaNs before calling predict_live_traffic().")
        else:
            print(f"    Rows with sufficient missingness exist but data_quality_risk=0.")
            print(f"    Check if NaN filling happened before predict_live_traffic was called.")

    print(f"\n  Simulated data_quality_risk counts at different tolerance values:\n")
    print(f"  {'tolerance':>10}  {'dq_risk=1':>10}  {'% of rows':>10}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}")
    for t in [0.05, 0.10, 0.15, 0.18, 0.20, 0.25, 0.30, tol]:
        t    = round(t, 4)
        n    = (missing_pct > t).sum()
        pct  = 100 * n / len(missing_pct)
        mark = "  ← CURRENT" if abs(t - tol) < 1e-4 else ""
        print(f"  {t:>10.4f}  {n:>10,}  {pct:>9.2f}%{mark}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PER-FEATURE MAD FLAG RATES
# Answers: which features are actually driving anomaly flags?
# ══════════════════════════════════════════════════════════════════════════════

def diag_feature_flag_rates(df_scores: pd.DataFrame) -> None:
    """
    Shows flag rate per feature across all live rows.
    If every feature has a 0% flag rate, your mad_threshold is too high for
    the live data's z-score distribution.
    If one feature dominates with 99% flag rate, it's likely a data artefact.
    """
    print("\n" + "═"*65)
    print("SECTION 5 — PER-FEATURE MAD FLAG RATES")
    print("═"*65)

    flag_cols = [c for c in df_scores.columns if c.endswith("_flag")
                 and c not in {"iso_forest_flag", "mad_critical_flag",
                               "data_quality_risk", "confirmed_threat",
                               "high_risk_review", "telemetry_gap_flag"}]

    if not flag_cols:
        print("\n  No per-feature flag columns found in df_scores.")
        print("  These are named '{feature}_flag' and require Critic fix [FIX-04].")
        return

    total = len(df_scores)
    rates = {col: df_scores[col].sum() / total for col in flag_cols}
    sorted_rates = sorted(rates.items(), key=lambda x: x[1], reverse=True)

    print(f"\n  {'Feature':<35} {'Flagged':>8}  {'Rate':>7}  Bar")
    print(f"  {'-'*35} {'-'*8}  {'-'*7}  {'-'*30}")

    for col, rate in sorted_rates:
        feature_name = col.replace("_flag", "")
        n   = int(df_scores[col].sum())
        bar = "█" * min(int(rate * 200), 30)
        print(f"  {feature_name:<35} {n:>8,}  {rate:>6.2%}  {bar}")

    # Dead feature check
    zero_rate = [c.replace("_flag","") for c, r in sorted_rates if r == 0.0]
    if zero_rate:
        print(f"\n  ⚠ Features with 0% flag rate (never anomalous in live data):")
        for f in zero_rate:
            print(f"    - {f}")
        print("    Consider: are these features constant in your live data?")
        print("    Check with: df_live['feature_name'].nunique()")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — USER COVERAGE ANALYSIS
# Answers: are any users being skipped by cold start?
# ══════════════════════════════════════════════════════════════════════════════

def diag_user_coverage(detector, df_scores: pd.DataFrame) -> None:
    """
    Shows how many live users have personal baselines vs falling back to global.
    Cold start users get global baseline scores which may be less sensitive.
    """
    print("\n" + "═"*65)
    print("SECTION 6 — USER COVERAGE ANALYSIS")
    print("═"*65)

    trained_users = set(detector.baseline_stats.keys())
    live_users    = set(df_scores.index.get_level_values("user").unique())

    known     = live_users & trained_users
    cold_start= live_users - trained_users

    print(f"\n  Trained users (personal baseline): {len(trained_users):,}")
    print(f"  Live users seen:                   {len(live_users):,}")
    print(f"    → With personal baseline:        {len(known):,}  ({100*len(known)/max(len(live_users),1):.1f}%)")
    print(f"    → Cold start (global baseline):  {len(cold_start):,}  ({100*len(cold_start)/max(len(live_users),1):.1f}%)")

    if len(cold_start) > 0.3 * len(live_users):
        print(f"\n  ⚠ Over 30% of live users are cold-start.")
        print(f"    Cold-start users use the global population baseline.")
        print(f"    Their anomaly scores are less personalised and may miss")
        print(f"    role-specific behaviour (e.g. sysadmins with high transfer volumes).")

    # Per-user threat rate
    if "confirmed_threat" in df_scores.columns:
        threat_by_user = (
            df_scores.groupby(level="user")["confirmed_threat"]
            .sum()
            .sort_values(ascending=False)
        )
        flagged_users = threat_by_user[threat_by_user > 0]
        print(f"\n  Users with at least one confirmed_threat: {len(flagged_users):,}")
        if len(flagged_users) > 0:
            print(f"\n  Top 10 users by confirmed_threat count:")
            print(f"  {'User':<30} {'Threat Days':>12}")
            print(f"  {'-'*30} {'-'*12}")
            for user, count in flagged_users.head(10).items():
                marker = "  [COLD START]" if user in cold_start else ""
                print(f"  {str(user):<30} {int(count):>12}{marker}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — RECOMMENDED THRESHOLD ADJUSTMENTS
# ══════════════════════════════════════════════════════════════════════════════

def diag_recommendations(detector, df_scores: pd.DataFrame) -> None:
    """
    Synthesises findings from all sections into concrete recommended actions.
    """
    print("\n" + "═"*65)
    print("SECTION 7 — RECOMMENDATIONS")
    print("═"*65)

    issues   = []
    actions  = []

    # Check each flag
    n_conf = df_scores.get("confirmed_threat",  pd.Series(0)).sum()
    n_rev  = df_scores.get("high_risk_review",  pd.Series(0)).sum()
    n_gap  = df_scores.get("telemetry_gap_flag", pd.Series(0)).sum()
    n_mad  = df_scores.get("mad_critical_flag",  pd.Series(0)).sum()
    n_iso  = df_scores.get("iso_forest_flag",    pd.Series(0)).sum()
    n_dq   = df_scores.get("data_quality_risk",  pd.Series(0)).sum()

    if n_mad == 0:
        issues.append("mad_critical_flag never fires")
        actions.append(
            f"Lower critical_flag_ratio (currently {detector.critical_flag_ratio:.4f}). "
            f"Try 0.05 or 0.10. Or lower mad_threshold (currently {detector.mad_threshold}) "
            f"to flag more features per row."
        )

    if n_iso == 0:
        issues.append("iso_forest_flag never fires")
        actions.append(
            f"Raise iso_threshold (currently {detector.iso_threshold:.6f}) — "
            f"set it to a higher (less negative) value to flag more rows. "
            f"Try the 5th percentile of live scores instead of training scores."
        )

    if n_dq == 0:
        issues.append("data_quality_risk never fires → high_risk_review and telemetry_gap_flag impossible")
        actions.append(
            f"Lower missing_data_tolerance (currently {detector.missing_data_tolerance}). "
            f"Try 0.05 (5% feature missingness triggers data quality flag). "
            f"Or check if NaNs are being filled upstream before predict_live_traffic."
        )

    if n_mad > 0 and n_iso > 0 and n_conf < 50:
        issues.append("Both models firing but very few confirmed threats")
        actions.append(
            "Both signals exist but rarely agree on the same row. "
            "Check Section 5 — one model may be flagging a different subset of users. "
            "Consider lowering both thresholds simultaneously and rerunning."
        )

    if not issues:
        print("\n  ✓ No obvious threshold issues detected.")
        print(f"  confirmed_threat={n_conf:,}  high_risk_review={n_rev:,}  "
              f"telemetry_gap_flag={n_gap:,}")
        print("  If counts are lower than expected, consider:")
        print("  → Is your live data representative of the threat period?")
        print("  → Does the CERT dataset label file align with your live parquet dates?")
        return

    print("\n  Issues detected:\n")
    for i, (issue, action) in enumerate(zip(issues, actions), 1):
        print(f"  [{i}] {issue}")
        print(f"      → {action}\n")

    print("  Quick threshold override for next run (no retraining needed):")
    print("  ─────────────────────────────────────────────────────────────")
    print("  detector2 = InsiderThreatDetector.load('your_model.pkl')")
    print("  detector2.critical_flag_ratio  = 0.05   # lower MAD bar")
    print("  detector2.iso_threshold        = -0.05  # raise ISO bar (less negative)")
    print("  detector2.missing_data_tolerance = 0.05 # lower quality bar")
    print("  df_scores2 = detector2.predict_live_traffic('live.parquet')")
    print("\n  Note: these overrides take effect immediately without retraining.")
    print("  Retrain only when you want the thresholds locked permanently.")


# ══════════════════════════════════════════════════════════════════════════════
# MASTER RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_full_diagnostic(
    detector,
    df_scores: pd.DataFrame,
    df_live_features: pd.DataFrame,
) -> None:
    """
    Runs all 7 diagnostic sections in sequence.

    Args:
        detector         : Fitted InsiderThreatDetector instance.
        df_scores        : Output of detector.predict_live_traffic().
        df_live_features : Raw live feature DataFrame (same index as df_scores).
    """
    print("\n" + "█"*65)
    print("  INSIDER THREAT DETECTOR — FULL DIAGNOSTIC REPORT")
    print(f"  Total live rows:    {len(df_scores):,}")
    print(f"  Model version:      {getattr(detector, 'model_version', 'unknown')}")
    print(f"  Trained users:      {len(detector.baseline_stats):,}")
    print("█"*65)

    diag_score_distributions(df_scores)
    diag_mad_threshold(detector, df_scores)
    diag_iso_threshold(detector, df_scores)
    diag_data_quality(detector, df_scores, df_live_features)
    diag_feature_flag_rates(df_scores)
    diag_user_coverage(detector, df_scores)
    diag_recommendations(detector, df_scores)

    print("\n" + "═"*65)
    print("  END OF DIAGNOSTIC REPORT")
    print("═"*65 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full diagnostic on InsiderThreatDetector output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",  required=True, help="Path to .pkl model file.")
    parser.add_argument("--live",   required=True, help="Path to live .parquet file.")
    parser.add_argument("--scores", default=None,  help="Path to pre-computed scores .parquet (optional).")

    try:
        idx  = sys.argv.index("--")
        args = parser.parse_args(sys.argv[idx + 1:])
    except ValueError:
        args = parser.parse_args()

    # Import here so the module is usable without src/ on the path
    sys.path.append(str(Path(__file__).parent))
    from src.detector import InsiderThreatDetector

    logger.info(f"Loading model from {args.model}...")
    detector = InsiderThreatDetector.load(args.model)

    logger.info(f"Loading live features from {args.live}...")
    df_live = InsiderThreatDetector._load_with_user_day_index(
        args.live, context="diagnostics"
    )

    if args.scores and Path(args.scores).exists():
        logger.info(f"Loading pre-computed scores from {args.scores}...")
        df_scores = pd.read_parquet(args.scores)
        # Restore MultiIndex if it was flattened on save
        if "user" in df_scores.columns and "activity_date" in df_scores.columns:
            df_scores = df_scores.set_index(["user", "activity_date"])
    else:
        logger.info("No pre-computed scores provided — running predict_live_traffic...")
        df_scores = detector.predict_live_traffic(args.live)

    run_full_diagnostic(detector, df_scores, df_live)