"""
InsiderThreatDetector — dual-model insider threat scoring engine.

Architecture:
    Model 1: MAD-based per-feature anomaly scoring (modified z-score,
             locked against historical baseline statistics).
    Model 2: Isolation Forest with auto-derived training threshold.
    Fusion: explicit security-hardened threat categories that distinguish
            clean-data confirmed threats from dirty-data high-risk reviews,
            and surface telemetry-gap IOCs separately.

Provenance:
    Each fit serializes the trained object with `joblib.dump(self, ...)`
    under a timestamped version string. `InsiderThreatDetector.load()`
    integrity-checks the pickle before returning it.

Fixes retained from earlier review rounds:
    `train_score_percentiles` stored at fit time, enabling
    `inference.py` to compute batch-stable calibrated risk scores.
    `contamination='auto'` (CERT base rate ≈ 0.02%, not 5%).
     internal MultiIndex (user, activity_date); no non-unique 'user'
     index ambiguity.
"""

import logging
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


class InsiderThreatDetector:
    """
    Dual-model ML pipeline for insider threat detection.

    Why contamination='auto':
        The IsolationForest `contamination` parameter is the assumed
        fraction of anomalies in training data. Using `0.05` (5%) is
        wildly miscalibrated for CERT r4.2 — that dataset has ~70
        malicious user-days out of ~330,000 (≈ 0.02%, 250× lower).
        With contamination=0.05 the model sets its decision threshold
        so that 5% of training data is "anomalous", which floods the
        SOC with false positives.
        `'auto'` defers to the internal path-length distribution. The
        actual cut-point is then auto-derived from training scores in
        fit_baseline() and stored as `self.iso_threshold`.

    Why MultiIndex (user, activity_date):
        The original `set_index('user')` produced a NON-UNIQUE index
        because each user has multiple days of data. Downstream
        `.loc[user]` was ambiguous (one row vs. many) and silently
        misbehaved on joins. `(user, activity_date)` is unique and
        makes the "one row = one user-day" semantics explicit.
    """

    def __init__(
            self, 
            mad_threshold: float = 3.5, 
            missing_data_tolerance: float = 0.2,
            iso_flag_percentile: float = 1.0   # Bottom 1% of training score
    ):
        self.mad_threshold = mad_threshold
        self.missing_data_tolerance = missing_data_tolerance
        self.iso_flag_percentile = iso_flag_percentile

        # contamination='auto' — see class docstring for justification
        self.iso_pipeline = Pipeline([
            ('imputer',  SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler',   RobustScaler()),
            ('detector', IsolationForest(
                contamination='auto',          
                random_state=42,
                n_estimators=100,
            )),
        ])

        self.baseline_stats: dict = {}
        self.is_fitted: bool = False
        self.model_version: str | None = None

    # ────────────────────────────────────────────────────────────────────
    # Deserialization with integrity checks
    # ────────────────────────────────────────────────────────────────────
    @classmethod
    def load(cls, model_path: str) -> "InsiderThreatDetector":
        """Loads a serialized detector and verifies its integrity."""
        logging.info(f"[load] Deserializing model from: {model_path}")

        try:
            obj = joblib.load(model_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"[load] Model file not found at: '{model_path}'. "
                f"Verify the artifact path or rerun fit_baseline()."
            )
        except Exception as e:
            raise RuntimeError(
                f"[load] Failed to deserialize model. "
                f"File may be corrupted or incompatible. Reason: {e}"
            )

        if not isinstance(obj, cls):
            raise TypeError(
                f"[load] Loaded object is type '{type(obj).__name__}', "
                f"expected 'InsiderThreatDetector'. Wrong file loaded."
            )
        if not obj.is_fitted:
            raise RuntimeError(
                "[load] Loaded model has is_fitted=False. "
                "This model was serialized before training completed."
            )
        if not obj.baseline_stats:
            raise RuntimeError(
                "[load] Loaded model has empty baseline_stats. "
                "MAD scoring will be non-functional. Retrain and re-serialize."
            )

        logging.info(
            f"[load] Model loaded | version={obj.model_version} | "
            f"features={len(obj.baseline_stats)} | "
            f"iso_threshold={getattr(obj, 'iso_threshold', 'MISSING — retrain recommended')}"
        )
        return obj

    # ────────────────────────────────────────────────────────────────────
    # Input validation gate
    # ────────────────────────────────────────────────────────────────────
    def _validate_input(self, df: pd.DataFrame, context: str) -> None:
        """
        Gate-check for all incoming DataFrames. Expects 'user' and
        'activity_date' to exist as either columns or MultiIndex levels.
        """
        if len(df) == 0:
            raise ValueError(
                f"[{context}] Empty DataFrame received. "
                f"Check if the upstream ETL pipeline produced output."
            )

        has_user_col   = 'user' in df.columns
        has_user_index = 'user' in (df.index.names or [])
        if not (has_user_col or has_user_index):
            raise ValueError(
                f"[{context}] Required identifier 'user' is missing from "
                f"both columns and index. "
                f"Columns: {list(df.columns)} | Index: {df.index.names}"
            )

        # Duplicate (user, activity_date) check (column-based only)
        if 'activity_date' in df.columns and has_user_col:
            if df.duplicated(subset=['user', 'activity_date']).any():
                raise ValueError(
                    f"[{context}] Duplicate (user, activity_date) entries. "
                    f"Check upstream SQL GROUP BY."
                )

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            raise ValueError(
                f"[{context}] No numeric feature columns found. "
                f"Columns: {list(df.columns)}"
            )

        # Feature drift (inference only)
        if self.is_fitted and context == "predict_live_traffic":
            trained = set(self.baseline_stats.keys())
            live    = set(numeric_cols)
            missing = trained - live
            new     = live - trained

            if missing:
                raise ValueError(
                    f"[{context}] FEATURE DRIFT — trained features missing from "
                    f"live data: {missing}. Retrain or fix ETL."
                )
            if new:
                logging.warning(
                    f"[{context}] New unseen columns in live data: {new}. "
                    "These will be ignored by the model."
                )

        logging.info(
            f"[{context}] Input validation passed. "
            f"Rows: {len(df)} | Numeric features: {len(numeric_cols)}"
        )

    # ────────────────────────────────────────────────────────────────────
    # Index helper — centralises the (user, activity_date) MultiIndex
    # construction so both fit and predict use the same logic.
    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def _load_with_user_day_index(parquet_path: str, context: str) -> pd.DataFrame:
        """
        Loads parquet and returns a DataFrame indexed by ('user', 'activity_date').
        Handles both cases:
            • parquet already has the MultiIndex (from fuse_feature_matrices)
            • parquet has 'user' and 'activity_date' as plain columns
        """
        try:
            df = pd.read_parquet(parquet_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"[{context}] Parquet file not found at: '{parquet_path}'. "
                f"Verify the path and check the upstream ETL job."
            )
        except Exception as e:
            raise RuntimeError(f"[{context}] Failed to load parquet. Reason: {e}")

        if list(df.index.names) == ['user', 'activity_date']:
            return df

        if 'user' in df.index.names:
            df = df.reset_index()
        if 'user' in df.columns and 'activity_date' in df.columns:
            return df.set_index(['user', 'activity_date'])

        raise ValueError(
            f"[{context}] Cannot construct (user, activity_date) MultiIndex. "
            f"Found columns: {list(df.columns)}, index: {df.index.names}"
        )

    # ────────────────────────────────────────────────────────────────────
    # TRAINING
    # ────────────────────────────────────────────────────────────────────
    def fit_baseline(self, historical_parquet_path: str):
        """Trains the model on clean historical data and serializes it to disk."""
        logging.info(f"TRAINING PHASE: Loading historical baseline from {historical_parquet_path}")

        # (user, activity_date) MultiIndex
        df_features = self._load_with_user_day_index(
            historical_parquet_path, context="fit_baseline"
        )

        # Validate against a reset-index view so column-based checks work
        self._validate_input(df_features.reset_index(), context="fit_baseline")

        # ── 1. MAD Baseline (Model 1) ───────────────────────────────────
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_median = df_features[col].median(skipna=True)
            col_mad    = median_abs_deviation(df_features[col].dropna(), scale='normal')
            col_p95    = df_features[col].quantile(0.95)   # NEW
            self.baseline_stats[col] = {'median': col_median, 'mad': col_mad, 'p95': col_p95}

        zero_mad_cols = [c for c, v in self.baseline_stats.items() if v['mad'] == 0]
        logging.info(
            f"[fit_baseline] MAD baseline computed | "
            f"features={len(self.baseline_stats)} | zero_mad_cols={zero_mad_cols}"
        )

        # ── 2. Isolation Forest (Model 2) ───────────────────────────────
        self.iso_pipeline.fit(df_features[numeric_cols])      
        self.is_fitted = True                               

        # Auto-derive ISO threshold from training distribution.
        # Was `mean - 1*std`, which flagged ~14% of training data — far too
        # noisy. Switched to a percentile cut (bottom 1% by default).
        # CERT r4.2 base rate is ~0.02% truly malicious, but the model also
        # flags legitimate-but-unusual behavior, so 1% leaves analyst
        # headroom without drowning the queue. Tune via the
        # `iso_flag_percentile` constructor arg.
        train_scores = self.iso_pipeline.decision_function(df_features[numeric_cols])
        self.iso_threshold = float(np.percentile(train_scores, self.iso_flag_percentile))
        logging.info(
            f"[fit_baseline] ISO threshold (percentile={self.iso_flag_percentile}) | "
            f"threshold={self.iso_threshold:.4f} | "
            f"score_range=[{train_scores.min():.4f}, {train_scores.max():.4f}]"
        )

        # ── C2: store training score percentiles for inference calibration ──   ← MISSING in your version
        quantile_points = np.linspace(0, 1, 1000)
        self.train_score_percentiles = list(
            np.quantile(train_scores, quantile_points).astype(float)
        )
        logging.info(
            f"[fit_baseline] Training score percentiles stored | "
            f"min={min(self.train_score_percentiles):.4f} | "
            f"p50={self.train_score_percentiles[500]:.4f} | "
            f"max={max(self.train_score_percentiles):.4f}"
        )

        # ── 3. Serialize ────────────────────────────────────────────────
        self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = f"iso_pipeline_v{self.model_version}.pkl"
        joblib.dump(self, export_path)
        logging.info(
            f"[fit_baseline] Model serialized | version={self.model_version} | "
            f"path={export_path} | features={list(self.baseline_stats.keys())}"
        )
        return export_path
        
    # ────────────────────────────────────────────────────────────────────
    # INFERENCE
    # ────────────────────────────────────────────────────────────────────
    def predict_live_traffic(
        self,
        live_parquet_path: str,
        iso_decision_threshold: float | None = None,
    ) -> pd.DataFrame:
        """
        Scores live traffic against the locked historical baseline.

        Returns a DataFrame indexed by (user, activity_date) containing:
            data_quality_risk     — 1 if >20% of features are NaN
            mad_score_count       — feature count exceeding MAD threshold
            mad_critical_flag     — 1 if mad_score_count >= 2
            iso_forest_raw_score  — continuous ISO Forest decision_function
            iso_forest_flag       — 1 if score < threshold
            confirmed_threat      — both models + clean data
            high_risk_review      — both models + dirty data
            data_loss_ioc         — telemetry gap only (no model signal)
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fit() before predict() is called. "
                "Run fit_baseline() first."
            )

        # Resolve threshold: caller override → auto-derived → fallback
        if iso_decision_threshold is None:
            if hasattr(self, 'iso_threshold'):
                iso_decision_threshold = self.iso_threshold
                logging.info(
                    f"[predict_live_traffic] Using auto-derived threshold: "
                    f"{iso_decision_threshold:.4f}"
                )
            else:
                iso_decision_threshold = -0.1
                logging.warning(
                    "[predict_live_traffic] iso_threshold missing. "
                    "Falling back to -0.1. Retrain to fix."
                )

        # MultiIndex (user, activity_date)
        df_features = self._load_with_user_day_index(
            live_parquet_path, context="predict_live_traffic"
        )

        self._validate_input(df_features.reset_index(), context="predict_live_traffic")

        logging.info(f"INFERENCE PHASE: Evaluating live traffic from {live_parquet_path}")

        # Scores DataFrame inherits the (user, activity_date) MultiIndex
        df_scores = pd.DataFrame(index=df_features.index)

        # ── NaN Audit (data_quality_risk IOC) ───────────────────────────
        missing_pct = df_features.isnull().mean(axis=1)
        df_scores['data_quality_risk'] = np.where(
            missing_pct > self.missing_data_tolerance, 1, 0
        )
        if df_scores['data_quality_risk'].sum() > 0:
            logging.warning(
                f"BLIND SPOT ALERT: {df_scores['data_quality_risk'].sum()} "
                f"user-days flagged for suspicious data loss."
            )

        # ── MODEL 1: MAD against locked baseline ────────────────────────
        mad_flags_matrix = pd.DataFrame(index=df_features.index)
        df_for_mad = df_features.fillna(0)

        for col in self.baseline_stats.keys():
            if col not in df_for_mad.columns:
                continue
            hist_median = self.baseline_stats[col]['median']
            hist_mad    = self.baseline_stats[col]['mad']
            live_values = df_for_mad[col]

            if hist_mad == 0:
                # MAD collapsed because >50% of training values equal the median.
                # But the feature may still have an informative non-zero tail.
                # Fall back to a percentile-based dispersion: the distance from the
                # median to the 95th percentile. If THAT is also 0, the feature is
                # genuinely constant and we abstain.
                hist_p95 = self.baseline_stats[col].get('p95', hist_median)
                fallback_scale = hist_p95 - hist_median
                if fallback_scale == 0:
                    # Genuinely constant feature (e.g. keyword_match_indicator,
                    # 100% zero in training) — no signal, abstain.
                    z_scores = np.zeros(len(live_values))
                else:
                    # Zero-inflated but informative (e.g. off_hours_ratio).
                    # Score live values by how far they exceed the normal envelope.
                    z_scores = np.clip(
                        (live_values - hist_median) / fallback_scale, 0, None
                    ) * self.mad_threshold  # scale so the threshold comparison is meaningful
            else:
                z_scores = np.abs(0.6745 * (live_values - hist_median) / hist_mad)

            mad_flags_matrix[f'{col}_flag'] = np.where(z_scores > self.mad_threshold, 1, 0)

        df_scores['mad_score_count']   = mad_flags_matrix.sum(axis=1)
        df_scores['mad_critical_flag'] = np.where(df_scores['mad_score_count'] >= 3, 1, 0)
        logging.info(
            f"[predict_live_traffic] MAD scoring complete | "
            f"critical_flags={df_scores['mad_critical_flag'].sum()} | "
            f"avg_flag_count={df_scores['mad_score_count'].mean():.2f}"
        )

        # ── MODEL 2: Isolation Forest ───────────────────────────────────
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        df_scores['iso_forest_raw_score'] = self.iso_pipeline.decision_function(
            df_features[numeric_cols]
        )
        df_scores['iso_forest_flag'] = np.where(
            df_scores['iso_forest_raw_score'] < iso_decision_threshold, 1, 0
        )
        logging.info(
            f"[predict_live_traffic] ISO Forest scoring complete | "
            f"range=[{df_scores['iso_forest_raw_score'].min():.4f}, "
            f"{df_scores['iso_forest_raw_score'].max():.4f}] | "
            f"threshold={iso_decision_threshold:.4f} | "
            f"flagged={df_scores['iso_forest_flag'].sum()}"
        )

        # ── DATA FUSION ─────────────────────────────────────────────────
        df_scores['confirmed_threat'] = np.where(
            (df_scores['mad_critical_flag'] == 1) &
            (df_scores['iso_forest_flag']   == 1) &
            (df_scores['data_quality_risk'] == 0),
            1, 0
        )
        df_scores['high_risk_review'] = np.where(
            (df_scores['mad_critical_flag'] == 1) &
            (df_scores['iso_forest_flag']   == 1) &
            (df_scores['data_quality_risk'] == 1),
            1, 0
        )
        df_scores['data_loss_ioc'] = np.where(
            (df_scores['data_quality_risk'] == 1) &
            (df_scores['confirmed_threat']  == 0) &
            (df_scores['high_risk_review']  == 0),
            1, 0
        )
        logging.info(
            f"[predict_live_traffic] Threat summary | "
            f"confirmed={df_scores['confirmed_threat'].sum()} | "
            f"high_risk_review={df_scores['high_risk_review'].sum()} | "
            f"data_loss_ioc={df_scores['data_loss_ioc'].sum()}"
        )

        return df_scores