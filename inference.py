import pandas as pd
import numpy as np
import logging
import shap
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from src.detector import InsiderThreatDetector

# ──────────────────────────────────────────────────────────────────────────────
# SHAP helper — loops over all trained features instead of hardcoding 3 names
# ──────────────────────────────────────────────────────────────────────────────
def _compute_shap_columns(
    model_pipeline,
    X: pd.DataFrame,
    feature_cols: list,
) -> pd.DataFrame:
    """
    Runs TreeExplainer on the IsolationForest inside the sklearn Pipeline.
    Returns a DataFrame of per-feature SHAP columns named shap_<feature>.

    Uses the *trained* model extracted from the pipeline — not a fresh one.
    SHAP values indicate how much each feature pushed a sample toward anomaly.
    Positive SHAP = feature increased anomaly score = increased risk.
    """
    logging.info("[SHAP] Computing feature attributions via TreeExplainer...")

    # Extract the fitted IsolationForest step from inside the sklearn Pipeline
    iso_step = model_pipeline.named_steps["detector"]

    try:
        explainer = shap.TreeExplainer(iso_step)
        # Transform X through the pipeline's pre-processing steps (imputer + scaler)
        # but stop before the final estimator — SHAP sees the same scaled space
        # the model was trained on, which is required for correct attributions.
        X_transformed = model_pipeline[:-1].transform(X)
        shap_values = explainer.shap_values(X_transformed)
    except Exception as e:
        logging.warning(
            f"[SHAP] TreeExplainer failed ({e}). "
            "Returning zero SHAP columns. Non-fatal — dashboard will show zeros."
        )
        shap_values = np.zeros((len(X), len(feature_cols)))

    shap_df = pd.DataFrame(
        shap_values,
        index=X.index,
        columns=[f"shap_{col}" for col in feature_cols],
    )
    logging.info(f"[SHAP] Done. Columns generated: {list(shap_df.columns)}")
    return shap_df


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────
def run_daily_inference(
    input_parquet_path: str,
    model_path: str,
    output_dir: str,
) -> str:
    """
    Executes the production ML inference pipeline.

    Steps
    -----
    1. Load the trained InsiderThreatDetector (.pkl serialised by fit_baseline).
    2. Run predict_live_traffic() — dual-model MAD + Isolation Forest scoring.
    3. Compute SHAP attributions from the *locked* trained model.
    4. Compute calibrated risk_score from training percentile distribution (C2 fix).
    5. Merge scores + SHAP + raw features into one flat DataFrame.
    6. Write scored_features_latest.parquet for the Streamlit dashboard.

    Parameters
    ----------
    input_parquet_path : str
        Path to the fused feature matrix produced by fuse_feature_matrices().
        Expected index: MultiIndex ['user', 'activity_date'].
    model_path : str
        Path to the .pkl produced by InsiderThreatDetector.fit_baseline().
    output_dir : str
        Directory where scored_features_latest.parquet is written.

    Returns
    -------
    str
        Absolute path to the output parquet.
    """
    logging.info("=" * 70)
    logging.info("PRODUCTION INFERENCE PIPELINE — START")
    logging.info(f"  features : {input_parquet_path}")
    logging.info(f"  model    : {model_path}")
    logging.info(f"  output   : {output_dir}")
    logging.info("=" * 70)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 1 — Load the trained detector
    # ──────────────────────────────────────────────────────────────────────
    # InsiderThreatDetector.load() validates:
    #   • is_fitted == True  (not a newly initialised, untrained object)
    #   • baseline_stats non-empty  (MAD model has data)
    #   • object type is InsiderThreatDetector  (wrong pkl caught)
    # Raises before touching any live data if any check fails.
    detector = InsiderThreatDetector.load(model_path)
    logging.info(
        f"[inference] Detector loaded | "
        f"version={detector.model_version} | "
        f"features={list(detector.baseline_stats.keys())}"
    )

    # ──────────────────────────────────────────────────────────────────────
    # STEP 2 — Dual-model scoring via predict_live_traffic()
    # ──────────────────────────────────────────────────────────────────────
    # predict_live_traffic() returns df_scores (index = user), containing:
    #
    #   data_quality_risk    — 1 if >20% of features are NaN (telemetry gap IOC)
    #   mad_score_count      — how many features exceeded the MAD threshold
    #   mad_critical_flag    — 1 if >= 2 features exceeded MAD threshold
    #   iso_forest_raw_score — continuous score from the trained Isolation Forest
    #   iso_forest_flag      — 1 if score < auto-derived training threshold
    #   confirmed_threat     — both models agree AND data is clean
    #   high_risk_review     — both models agree BUT data is suspicious
    #   data_loss_ioc        — data quality alone is suspicious (no model signal)
    #
    # predict_live_traffic() does NOT mutate the raw feature DataFrame.
    df_scores = detector.predict_live_traffic(input_parquet_path)
    logging.info(
        f"[inference] Scoring complete | "
        f"rows={len(df_scores)} | "
        f"confirmed_threats={df_scores['confirmed_threat'].sum()} | "
        f"high_risk_reviews={df_scores['high_risk_review'].sum()} | "
        f"data_loss_iocs={df_scores['data_loss_ioc'].sum()}"
    )

    # ──────────────────────────────────────────────────────────────────────
    # STEP 3 — Reload raw features for downstream merge
    # ──────────────────────────────────────────────────────────────────────
    # predict_live_traffic() intentionally returns ONLY scores (immutable
    # design). We reload the features here for the dashboard merge.
    df_raw = pd.read_parquet(input_parquet_path)
    if "user" in df_raw.index.names:
        df_raw = df_raw.reset_index()

    # ──────────────────────────────────────────────────────────────────────
    # STEP 4 — SHAP attributions from the LOCKED trained model
    # ──────────────────────────────────────────────────────────────────────
    # Critical: we score SHAP against detector.iso_pipeline (the trained
    # pipeline), NOT a freshly fit model. This guarantees:
    #   • Explanations reflect what the deployed model learned from history.
    #   • SHAP values are consistent and comparable across daily batches.
    #   • No new model is ever fit on live data during inference.
    feature_cols = list(detector.baseline_stats.keys())

    # Build the feature matrix identically to how predict_live_traffic does it
    df_for_shap = df_raw.copy()
    # H8: build the same (user, activity_date) MultiIndex the detector uses internally,
    # so shap_df aligns with df_scores for the .join() below.
    if list(df_for_shap.index.names) != ["user", "activity_date"]:
        if "user" in df_for_shap.index.names:
            df_for_shap = df_for_shap.reset_index()
    df_for_shap = df_for_shap.set_index(["user", "activity_date"])
    X_live = (
        df_for_shap
        .select_dtypes(include=[np.number])
        [feature_cols]
        .fillna(0)
    )

    shap_df = _compute_shap_columns(
        model_pipeline=detector.iso_pipeline,
        X=X_live,
        feature_cols=feature_cols,
    )

    # ──────────────────────────────────────────────────────────────────────
    # STEP 5 — Calibrated risk_score from training score distribution
    # ──────────────────────────────────────────────────────────────────────
    # C2 fix: risk_score is a PERCENTILE on the TRAINING score distribution,
    # not a batch-relative normalisation.
    #
    #   risk_score = 0.95  →  more anomalous than 95% of training-era users
    #   risk_score = 0.50  →  median normality
    #   risk_score = 0.10  →  clearly normal
    #
    # This makes the 0.75 critical threshold STABLE and MEANINGFUL across
    # batches. A boring day will NOT auto-produce a user with risk_score=1.0.
    #
    # Requires detector.train_score_percentiles (stored by fit_baseline after
    # this fix is applied). For models trained before this fix, falls back
    # to a safe batch-relative method with an explicit warning to retrain.
    raw_scores = df_scores["iso_forest_raw_score"].values

    if hasattr(detector, "train_score_percentiles"):
        # Empirical CDF: map each live score to its percentile on training scores.
        # ISO scores: higher = more normal → we invert so higher percentile = more risk.
        percentiles = np.sort(np.array(detector.train_score_percentiles))
        n = len(percentiles)
        # For each live score, find what fraction of training scores are HIGHER (more normal)
        # i.e., how anomalous is this score relative to training?
        risk_score = np.array([
            float(np.searchsorted(percentiles, s, side='right')) / n
            for s in raw_scores
        ])
        # Invert: low iso score (anomalous) → high percentile rank → high risk
        risk_score = 1.0 - risk_score
        risk_score = np.clip(risk_score, 0.0, 1.0)
        logging.info(
            "[inference] risk_score calibrated against training percentiles. "
            f"Range: [{risk_score.min():.3f}, {risk_score.max():.3f}]"
        )
    else:
        logging.warning(
            "[inference] train_score_percentiles not found on loaded model. "
            "This model was trained before the C2 calibration fix. "
            "Falling back to batch-relative normalisation. "
            "Retrain with updated fit_baseline() to resolve this."
        )
        score_range = raw_scores.max() - raw_scores.min()
        if score_range == 0:
            logging.warning(
                "[inference] All ISO scores identical — "
                "degenerate batch. risk_score set to 0.5 for all users."
            )
            risk_score = np.full(len(raw_scores), 0.5)
        else:
            risk_score = (raw_scores.max() - raw_scores) / score_range

    df_scores["risk_score"] = risk_score

    # ──────────────────────────────────────────────────────────────────────
    # STEP 6 — Merge scores + SHAP + raw features into one flat DataFrame
    # ──────────────────────────────────────────────────────────────────────
    # H8: df_scores is now indexed by (user, activity_date), as is shap_df.
    # Join everything ON the MultiIndex — safe, deterministic, no duplicate cols.
    df_scores = df_scores.join(shap_df, how="left")

    # Attach raw features (already indexed by (user, activity_date) from
    # fuse_feature_matrices). Index alignment guarantees correct row matching
    # regardless of row order in the source parquets.
    df_raw_indexed = df_for_shap[[c for c in feature_cols if c in df_for_shap.columns]]
    df_final = df_scores.join(df_raw_indexed, how="left")

    # Reset index so user and activity_date become plain columns for the parquet
    df_final = df_final.reset_index()

    
    # ──────────────────────────────────────────────────────────────────────
    # STEP 7 — Export for dashboard
    # ──────────────────────────────────────────────────────────────────────
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    final_output_file = out_path / "scored_features_latest.parquet"

    df_final.to_parquet(final_output_file, index=False)
    logging.info(f"[inference] Output written → {final_output_file}")

    # Operator summary
    critical_count  = int((df_final["risk_score"] > 0.75).sum())
    confirmed       = int(df_final["confirmed_threat"].sum())
    review          = int(df_final["high_risk_review"].sum())
    ioc             = int(df_final["data_loss_ioc"].sum())

    logging.info("=" * 70)
    logging.info("INFERENCE SUMMARY")
    logging.info(f"  User-days scored                              : {len(df_final)}")
    logging.info(f"  risk_score > 0.75 (calibrated percentile)    : {critical_count}")
    logging.info(f"  confirmed_threat  (both models + clean data)  : {confirmed}")
    logging.info(f"  high_risk_review  (both models + dirty data)  : {review}")
    logging.info(f"  data_loss_ioc     (telemetry gap only)        : {ioc}")
    logging.info("=" * 70)

    return str(final_output_file)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # input_parquet_path — the fused live feature matrix from fuse_feature_matrices()
    # model_path         — the .pkl saved by fit_baseline()
    # Update both paths to match your most recent run before executing.
    run_daily_inference(
        input_parquet_path="features/live_test_20260516_001301.parquet",
        model_path="iso_pipeline_v20260516_010900.pkl",
        output_dir="features/",
    )