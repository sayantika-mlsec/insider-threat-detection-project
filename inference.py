import pandas as pd
import numpy as np
import logging
import joblib
import shap
from pathlib import Path
from sklearn.ensemble import IsolationForest

# Configure standard logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_daily_inference(input_parquet_path: str, model_path: str, output_dir: str):
    """
    Executes the ML inference pipeline: anomaly scoring + SHAP explanations.
    """
    logging.info(f"Starting ML Inference Pipeline...")
    logging.info(f"Reading target features from: {input_parquet_path}")

    # =====================================================================
    # 1. LOAD DATA & VALIDATE
    # =====================================================================
    try:
        df_raw = pd.read_parquet(input_parquet_path)
    except FileNotFoundError:
        logging.error("DuckDB feature matrix not found. Run extraction first.")
        raise

    # For inference, we separate the metadata (Index) from the mathematical features
    # df_raw index is currently MultiIndex: ['user', 'activity_date']
    df_features = df_raw.copy()

    # The features we want the model to actually evaluate
    target_columns = [
        'total_events', 'off_hours_ratio', 'unique_systems_accessed', 
        'failed_login_ratio', 'active_orphan_sessions'
    ]
    
    # Ensure all target columns exist (handling the NaN schema alignment)
    X = df_features[target_columns].fillna(0) # Zero imputation for missing daily data

    # =====================================================================
    # 2. MODEL INFERENCE (Anomaly Scoring)
    # =====================================================================
    # In a true production environment, you would load your saved .pkl model here:
    # detector = InsiderThreatDetector.load(model_path)
    # df_scores = detector.predict_live_traffic(df_features)
    
    # For this MVP to run instantly, we will initialize and fit a fresh model 
    # directly on the incoming data stream to act as a dynamic baseline.
    logging.info("Initializing dynamic Isolation Forest baseline...")
    
    iso_forest = IsolationForest(
        n_estimators=150, 
        contamination=0.05, # Assuming top 5% of daily activity is highly anomalous
        random_state=42
    )
    
    iso_forest.fit(X)
    
    # decision_function returns negative values for anomalies, positive for normal.
    # We invert it (multiply by -1) and scale it so 0 = Normal, 1.0 = Critical Threat.
    # This makes it readable for the SOC Dashboard.
    raw_scores = iso_forest.decision_function(X)
    normalized_risk = (raw_scores.max() - raw_scores) / (raw_scores.max() - raw_scores.min())
    
    df_features['risk_score'] = normalized_risk

    # =====================================================================
    # 3. EXPLAINABLE AI (SHAP)
    # =====================================================================
    logging.info("Calculating SHAP Feature Attributions...")
    
    # SHAP TreeExplainer is heavily optimized for Isolation Forests
    explainer = shap.TreeExplainer(iso_forest)
    shap_values = explainer.shap_values(X)
    
    # IsolationForest SHAP values are structured slightly differently.
    # We want to know how much each feature pushed the user TOWARDS an anomaly.
    # We append these SHAP values directly back into the dataframe.
    df_features['shap_off_hours'] = shap_values[:, X.columns.get_loc('off_hours_ratio')]
    df_features['shap_orphans'] = shap_values[:, X.columns.get_loc('active_orphan_sessions')]
    df_features['shap_failed_logins'] = shap_values[:, X.columns.get_loc('failed_login_ratio')]

    # =====================================================================
    # 4. DATA FUSION & TELEMETRY CHECKS
    # =====================================================================
    logging.info("Applying Data Quality Fusion rules...")
    
    # Check for missing telemetry (the NaN audit we discussed earlier)
    # If a user has 0 total_events but is in the system, logs might be severed
    df_features['data_quality_risk'] = np.where(df_features['total_events'] == 0, 1, 0)

    # =====================================================================
    # 5. EXPORT FOR DASHBOARD
    # =====================================================================
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    final_output_file = out_path / "scored_features_latest.parquet"
    
    # Save the fully scored dataframe (including the index)
    df_features.to_parquet(final_output_file, index=True)
    
    logging.info(f"Pipeline Complete. Scored data written to: {final_output_file}")
    
    # Print a quick summary to the console
    critical_count = len(df_features[df_features['risk_score'] > 0.75])
    logging.info(f"Found {critical_count} critical anomalies in this batch.")

if __name__ == "__main__":
    run_daily_inference(
        # Point this to the 'live' test file DuckDB just generated
        input_parquet_path="features/live_traffic_test_20260503_002624.parquet", 
        
        # Point this to your saved model from the fit_baseline step
        model_path="iso_pipeline_v20260503_003905.pkl",                 
        
        output_dir="features/"                                       
    )