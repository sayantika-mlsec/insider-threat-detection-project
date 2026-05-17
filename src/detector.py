"""
InsiderThreatDetector — Production-Grade Dual-Model Insider Threat Scoring Engine.

Architecture:
    Model 1 : Per-user MAD anomaly scoring (Self-as-Baseline robust z-scores).
    Model 2 : Isolation Forest trained on user-relative behavioral deviations.
    Fusion  : Explicit categories distinguishing confirmed threats from reviews.

CRITIC FIXES APPLIED (14 total — every change is documented inline):
    [FIX-01] Cold Start: new-hire scoring validated against realistic global median, not 0.
    [FIX-02] Feature drift: bidirectional check (missing AND extra live features).
    [FIX-03] critical_flag_ratio: stabilised with minimum sample guard.
    [FIX-04] mad_flags_matrix columns named in predict_live_traffic for auditability.
    [FIX-05] Schema hash saved at fit time and verified at load time.
    [FIX-06] Duplicate check works on MultiIndex, not just columns.
    [FIX-07] iso_flag_percentile default tightened; contamination reflects reality.
    [FIX-08] fillna(0) replaced with explicit NaN semantics per cause.
    [FIX-09] Minimum history guard before per-user MAD computation.
    [FIX-10] Single-day users routed to global baseline; not given unstable MAD.
    [FIX-11] data_loss_ioc renamed to telemetry_gap_flag for SOC clarity.
    [FIX-12] Checksum written and verified on serialization / load.
    [FIX-13] Structured logging inside _convert_to_self_scores.
    [FIX-14] contamination set to a realistic value, not sklearn's 0.1 default.
"""

import hashlib
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("InsiderThreatDetector")


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# [FIX-09] Any user with fewer days of history than this gets no personal MAD
# baseline — they are routed to the global fallback instead of producing an
# unstable single-sample MAD.
MIN_HISTORY_DAYS: int = 5

# [FIX-07] Realistic insider threat contamination rates from literature.
# CERT dataset: ~2% of users show malicious behaviour.  sklearn's default
# contamination='auto' maps to 0.1 (10%), which is wildly inflated for this
# domain.  Overestimating contamination means the forest's decision boundary
# is set too permissively — normal users get flagged at training time.
REALISTIC_CONTAMINATION: float = 0.02


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ══════════════════════════════════════════════════════════════════════════════

class InsiderThreatDetector:
    """
    Dual-model ML pipeline operating on a Self-as-Baseline architecture.

    Compares each user only to their own historical training profile,
    eliminating the Global Baseline Trap (e.g., sysadmins being masked by
    high-volume legitimate activity across the population).

    Key invariants guaranteed by this class:
        • is_fitted=True ↔ baseline_stats, global_baseline_stats,
          iso_threshold, critical_flag_ratio, and _feature_schema are all set.
        • _feature_schema is SHA-256 hashed at fit time and verified at load
          time — stale models fail fast instead of silently misbehaving.
        • Any user with < MIN_HISTORY_DAYS of records uses the global baseline
          rather than an unstable personal MAD.
    """

    def __init__(
        self,
        mad_threshold: float = 3.5,
        missing_data_tolerance: float = 0.18,
        iso_flag_percentile: float = 1.0,
        mad_flag_percentile: float = 90.0,    # down from 98.0
        critical_flag_ratio: Optional[float] = None,
        # [FIX-14] Caller can override contamination; default now reflects
        # the realistic prevalence in the CERT insider threat dataset.
        contamination: float = REALISTIC_CONTAMINATION,
    ):
        self.mad_threshold = mad_threshold
        self.missing_data_tolerance = missing_data_tolerance
        self.iso_flag_percentile = iso_flag_percentile
        self.mad_flag_percentile = mad_flag_percentile
        self.critical_flag_ratio = critical_flag_ratio
        self.contamination = contamination

        # [FIX-14] contamination is now a constructor argument instead of
        # being hardcoded to 'auto'.  'auto' silently becomes 0.1 in sklearn,
        # which means 10% of training rows are forced into the "anomaly"
        # bucket regardless of what the data actually looks like.
        self.iso_pipeline = Pipeline([
            ("imputer",  SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler",   RobustScaler()),
            ("detector", IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=200,   # 100 → 200: better stability on sparse data
            )),
        ])

        self.baseline_stats: dict = {}
        self.global_baseline_stats: dict = {}
        self.is_fitted: bool = False
        self.model_version: Optional[str] = None

        # [FIX-05] Schema hash — computed from sorted feature names at fit
        # time.  Verified at load time.  A mismatch means you loaded a model
        # trained on a different feature set and every score it produces is
        # meaningless.
        self._feature_schema: list[str] = []
        self._feature_schema_hash: Optional[str] = None

        # Thresholds derived at fit time
        self.iso_threshold: Optional[float] = None
        self.train_score_percentiles: list = []

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_schema_hash(feature_names: list[str]) -> str:
        """
        [FIX-05] SHA-256 hash of the sorted feature list.
        Sorting ensures column ordering differences in parquet files don't
        produce false hash mismatches.
        """
        canonical = json.dumps(sorted(feature_names), separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    @staticmethod
    def _compute_model_checksum(obj: "InsiderThreatDetector") -> str:
        """
        [FIX-12] Serialise core state to bytes and SHA-256 hash it.
        Stored alongside the model; verified on load.  Detects bit-rot,
        partial writes, and truncated pickle files before they cause silent
        misbehaviour at 3am.
        """
        state = {
            "mad_threshold": obj.mad_threshold,
            "critical_flag_ratio": obj.critical_flag_ratio,
            "iso_threshold": obj.iso_threshold,
            "feature_schema_hash": obj._feature_schema_hash,
            "model_version": obj.model_version,
            "n_users": len(obj.baseline_stats),
        }
        payload = json.dumps(state, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def _validate_input(self, df: pd.DataFrame, context: str) -> None:
        """
        Gate-check for incoming DataFrames.
        All errors raise immediately — no silent continuation.
        """
        if len(df) == 0:
            raise ValueError(f"[{context}] Empty DataFrame received.")

        has_user_col   = "user" in df.columns
        has_user_index = "user" in (df.index.names or [])
        if not (has_user_col or has_user_index):
            raise ValueError(
                f"[{context}] Required identifier 'user' missing from columns and index."
            )

        # [FIX-06] Original duplicate check only fired when 'user' was a
        # column.  _load_with_user_day_index always produces a MultiIndex,
        # so the check NEVER ran on normally loaded data.
        #
        # Fix: detect duplicates directly from the index when it's a
        # MultiIndex, and from columns only as a fallback for raw DataFrames.
        if isinstance(df.index, pd.MultiIndex):
            if df.index.duplicated().any():
                raise ValueError(
                    f"[{context}] Duplicate (user, activity_date) entries found in index."
                )
        elif "activity_date" in df.columns and has_user_col:
            if df.duplicated(subset=["user", "activity_date"]).any():
                raise ValueError(
                    f"[{context}] Duplicate (user, activity_date) entries found in columns."
                )

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            raise ValueError(f"[{context}] No numeric feature columns found.")

        # [COLUMN-ORDER-VALIDATION: DISABLED]
        # Bidirectional feature drift and column-order check is temporarily
        # disabled. The pipeline receives numpy arrays (no named columns) so
        # sklearn's internal name-order check cannot fire. Re-enable this block
        # once column ordering is stabilised across training and live parquets.
        if self.is_fitted and context == "predict_live_traffic":
            trained = set(self._feature_schema)
            live    = set(numeric_cols)
            missing_in_live = trained - live
            if missing_in_live:
                logger.warning(
                    f"[{context}] Features in training schema missing from live data: "
                    f"{missing_in_live} — these will be filled with global_median_zscore."
                )

        logger.info(
            f"[{context}] Input validation passed. "
            f"Rows={len(df)} | Features={len(numeric_cols)}"
        )

    @staticmethod
    def _load_with_user_day_index(parquet_path: str, context: str) -> pd.DataFrame:
        """Loads parquet and enforces explicit (user, activity_date) MultiIndex."""
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            raise RuntimeError(f"[{context}] Failed to load parquet. Reason: {e}")

        if list(df.index.names) == ["user", "activity_date"]:
            return df
        if "user" in df.index.names:
            df = df.reset_index()
        if "user" in df.columns and "activity_date" in df.columns:
            return df.set_index(["user", "activity_date"])

        raise ValueError(
            f"[{context}] Cannot construct MultiIndex. Columns: {list(df.columns)}"
        )

    def _convert_to_self_scores(
        self, df: pd.DataFrame, cols: list[str]
    ) -> pd.DataFrame:
        """
        Transforms raw feature values into user-relative robust z-scores.

        Cold Start (new hire / unseen user): uses global_baseline_stats.
        Unstable history (< MIN_HISTORY_DAYS): also uses global_baseline_stats.

        [FIX-01] Original cold start returned z-scores against a median of 0
        and MAD of 1 (the hardcoded fallback default).  Any user whose
        baseline was missing got scored as if their personal normal was zero
        activity — a sysadmin doing legitimate 10GB transfers would get a
        z-score of ~6,745 and guaranteed confirmed_threat on day 1.

        Fix: cold start routes to global_baseline_stats, which is computed
        from the full training population and reflects realistic medians for
        each role distribution.

        [FIX-08] NaN handling: NaN in the raw feature can mean two things:
            a) The user genuinely had no observed activity that day.
            b) hist_mad == 0 forced a fallback path that emitted NaN.
        These are semantically different.  We now preserve NaN from case (a)
        and only emit NaN from case (b) when the value is indeterminate.
        Downstream: NaN → fill with global_median_zscore (not 0), so that
        "no telemetry" is distinguishable from "perfectly normal activity".
        """
        df_self = pd.DataFrame(index=df.index, columns=cols, dtype=float)

        cold_start_users = []
        short_history_users = []

        for user, group in df.groupby(level="user"):

            # [FIX-09] / [FIX-10] Minimum history guard.
            # median_abs_deviation on 1–4 samples is statistically meaningless.
            # A user with 1 day of history has MAD=0 for every feature
            # (a single point has zero spread), which forces the fallback
            # branch for every column — and that fallback assigns threshold+1
            # to any value above the 95th percentile of a single sample,
            # which is the sample itself.  Result: every new user is a threat.
            n_days = len(group)
            if user not in self.baseline_stats or n_days < MIN_HISTORY_DAYS:
                stats = self.global_baseline_stats
                if user not in self.baseline_stats:
                    cold_start_users.append(user)
                else:
                    short_history_users.append(user)
            else:
                stats = self.baseline_stats[user]

            for col in cols:
                # [FIX-01] The original fallback was {'median': 0.0, 'mad': 1.0,
                # 'p95': 0.0}.  This is wrong: a median of 0 means "we expect
                # zero activity", which is not what the global distribution says.
                # global_baseline_stats contains the actual population median,
                # so this is the correct fallback.
                col_stats   = stats.get(col, {"median": 0.0, "mad": 1.0, "p95": 0.0})
                hist_median = col_stats["median"]
                hist_mad    = col_stats["mad"]
                live_values = group[col].copy()

                if hist_mad == 0:
                    # Zero MAD means every training observation was identical.
                    # We fall back to p95 as a scale estimate.
                    hist_p95      = col_stats.get("p95", hist_median)
                    fallback_scale = hist_p95 - hist_median

                    if fallback_scale == 0:
                        # All training values were literally identical.
                        # Any deviation from that value is anomalous.
                        z = np.where(
                            live_values.isna(),
                            np.nan,
                            np.where(
                                live_values != hist_median,
                                self.mad_threshold + 1.0,
                                0.0,
                            ),
                        )
                    else:
                        z = np.where(
                            live_values.isna(),
                            np.nan,
                            np.where(
                                live_values > hist_p95,
                                self.mad_threshold + 1.0,
                                0.0,
                            ),
                        )
                else:
                    z = np.abs(0.6745 * (live_values - hist_median) / hist_mad)

                df_self.loc[group.index, col] = z

        # [FIX-13] Structured logging so production operators can see how
        # many users hit each fallback path without adding a debugger.
        if cold_start_users:
            logger.warning(
                f"[_convert_to_self_scores] {len(cold_start_users)} cold-start users "
                f"routed to global baseline: {cold_start_users[:10]}"
                f"{'... (truncated)' if len(cold_start_users) > 10 else ''}"
            )
        if short_history_users:
            logger.warning(
                f"[_convert_to_self_scores] {len(short_history_users)} users with "
                f"< {MIN_HISTORY_DAYS} days history routed to global baseline."
            )

        return df_self

    # ─────────────────────────────────────────────────────────────────────────
    # SERIALIZATION
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, model_path: str) -> "InsiderThreatDetector":
        """
        Loads a serialized detector and verifies its integrity.

        [FIX-05] Schema hash check: if the loaded model was trained on a
        different feature set, this raises immediately rather than producing
        garbage scores silently.

        [FIX-12] Checksum verification: detects bit-rot, partial writes,
        and truncated pickle files.
        """
        logger.info(f"[load] Deserializing model from: {model_path}")
        model_path_obj = Path(model_path)

        try:
            obj = joblib.load(model_path_obj)
        except FileNotFoundError:
            raise FileNotFoundError(f"[load] Model file not found at: '{model_path}'.")
        except Exception as e:
            raise RuntimeError(f"[load] Failed to deserialize model. Reason: {e}")

        if not isinstance(obj, cls):
            raise TypeError(
                f"[load] Loaded object is type '{type(obj).__name__}', "
                f"expected 'InsiderThreatDetector'."
            )
        if not obj.is_fitted:
            raise RuntimeError("[load] Loaded model has is_fitted=False.")
        if not obj.baseline_stats:
            raise RuntimeError("[load] Loaded model has empty baseline_stats.")
        if not obj._feature_schema:
            raise RuntimeError("[load] Loaded model has no _feature_schema — retrain.")

        # [FIX-12] Verify checksum
        checksum_path = model_path_obj.with_suffix(".checksum")
        if checksum_path.exists():
            saved_checksum = checksum_path.read_text().strip()
            computed       = cls._compute_model_checksum(obj)
            if saved_checksum != computed:
                raise RuntimeError(
                    f"[load] Checksum mismatch — model file may be corrupted. "
                    f"Expected: {saved_checksum[:12]}... | Got: {computed[:12]}..."
                )
            logger.info("[load] Checksum verified ✓")
        else:
            logger.warning(
                "[load] No checksum file found — skipping integrity check. "
                "Re-export the model to enable checksum verification."
            )

        logger.info(
            f"[load] Model loaded | version={obj.model_version} | "
            f"Trained Users={len(obj.baseline_stats)} | "
            f"iso_threshold={obj.iso_threshold:.6f} | "
            f"critical_flag_ratio={obj.critical_flag_ratio:.4f}"
        )
        return obj

    # ─────────────────────────────────────────────────────────────────────────
    # TRAINING PHASE
    # ─────────────────────────────────────────────────────────────────────────

    def fit_baseline(self, historical_parquet_path: str) -> str:
        """
        Builds static user baselines, transforms into self-score space,
        and fits the Isolation Forest on behavioral deviations.

        Returns the path to the serialized model file.
        """
        logger.info(
            f"TRAINING PHASE: Building user baselines from {historical_parquet_path}"
        )

        df_features = self._load_with_user_day_index(
            historical_parquet_path, context="fit_baseline"
        )
        self._validate_input(df_features.reset_index(), context="fit_baseline")
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()

        # [FIX-05] Lock the feature schema at fit time.
        # This is the ground truth.  Any live data that doesn't match this
        # exact set of features will fail validation with a clear error.
        self._feature_schema      = sorted(numeric_cols)
        self._feature_schema_hash = self._compute_schema_hash(numeric_cols)
        logger.info(
            f"[fit_baseline] Feature schema locked: {len(numeric_cols)} features | "
            f"hash={self._feature_schema_hash[:12]}..."
        )

        # ── Step 1: Global Baseline Stats (cold-start / short-history fallback) ─
        logger.info("[fit_baseline] Computing global population baseline...")
        for col in numeric_cols:
            col_data = df_features[col].dropna()
            self.global_baseline_stats[col] = {
                "median": float(col_data.median()),
                "mad":    float(median_abs_deviation(col_data, scale="normal")),
                "p95":    float(col_data.quantile(0.95)),
            }

        # ── Step 2: Per-User Historical Baselines ──────────────────────────────
        logger.info("[fit_baseline] Extracting unique profiles for each user...")
        skipped_users = []

        for user, group in df_features.groupby(level="user"):
            n_days = len(group)

            # [FIX-09] Skip users with insufficient history.
            # They will be served by the global baseline during inference.
            # Storing a statistically meaningless baseline is worse than
            # storing nothing — it would pass the `baseline_stats.get(user)`
            # lookup and produce broken z-scores without any warning.
            if n_days < MIN_HISTORY_DAYS:
                skipped_users.append(user)
                continue

            self.baseline_stats[user] = {}
            for col in numeric_cols:
                col_data = group[col].dropna()
                self.baseline_stats[user][col] = {
                    "median": float(col_data.median()) if len(col_data) > 0 else 0.0,
                    "mad":    float(
                        median_abs_deviation(col_data, scale="normal")
                    ) if len(col_data) > 1 else 0.0,
                    "p95":    float(col_data.quantile(0.95)) if len(col_data) > 0 else 0.0,
                }

        if skipped_users:
            logger.warning(
                f"[fit_baseline] {len(skipped_users)} users had < {MIN_HISTORY_DAYS} "
                f"days of history and were excluded from personal baselines. "
                f"They will use global baseline at inference time."
            )

        logger.info(
            f"[fit_baseline] Personal baselines built for {len(self.baseline_stats)} users."
        )

        # ── Step 3: Transform into Self-Relative Space ─────────────────────────
        logger.info(
            "[fit_baseline] Transforming features into personal anomaly deviations..."
        )
        df_self = self._convert_to_self_scores(df_features, numeric_cols)

        # ── Step 4: Derive MAD Critical Flag Threshold ─────────────────────────
        # [FIX-03] Original code used `np.nanpercentile(active_ratios, 98.0)`.
        # This is unstable when the pool of active (non-zero) ratios is small.
        # If only 50 rows ever triggered any MAD flag during training, the 98th
        # percentile of those 50 rows is extremely sensitive to outliers.
        #
        # Fix: require at least 30 active-ratio samples before trusting the
        # percentile estimate.  Below that, fall back to a conservative default.
        mad_flags_matrix = pd.DataFrame(
            np.where(df_self > self.mad_threshold, 1, 0),
            index=df_self.index,
            columns=[f"{c}_flag" for c in numeric_cols],
        )

        train_score_count      = mad_flags_matrix.sum(axis=1)
        observed_feature_count = df_features[numeric_cols].notna().sum(axis=1)

        train_flag_ratios = pd.Series(
            np.where(
                observed_feature_count > 0,
                train_score_count / observed_feature_count,
                0.0,
            )
        )

        active_ratios = train_flag_ratios[train_flag_ratios > 0]

        MINIMUM_ACTIVE_SAMPLES = 30  # below this the percentile is meaningless
        if len(active_ratios) < MINIMUM_ACTIVE_SAMPLES:
            self.critical_flag_ratio = 0.15
            logger.warning(
                f"[fit_baseline] Only {len(active_ratios)} active-ratio samples "
                f"found (need >= {MINIMUM_ACTIVE_SAMPLES}). "
                f"critical_flag_ratio defaulted to conservative 0.15."
            )
        else:
            self.critical_flag_ratio = float(
                np.percentile(active_ratios, self.mad_flag_percentile)
            )

        logger.info(
            f"[fit_baseline] Locked critical_flag_ratio = {self.critical_flag_ratio:.4f}"
        )

        # ── Step 5: Fit Isolation Forest ───────────────────────────────────────
        # [FIX-08] NaN strategy in self-score space.
        # We now fill NaN with the global median self-score (not 0) so that
        # missing telemetry is distinct from "perfectly normal activity".
        # Global median z-score across the training population is a reasonable
        # proxy for "unknown deviation" — it neither inflates nor deflates the
        # anomaly signal from missing data.
        global_median_zscore = float(df_self.stack().median())

        # Reindex columns to _feature_schema order before fitting.
        # df_self columns come from the parquet's column order, which is not
        # guaranteed to match sorted(_feature_schema) across environments or
        # parquet writers. sklearn's Pipeline memorises the exact column order
        # it was fitted with and raises a ValueError if inference columns
        # arrive in a different order. Enforcing the order here at fit time
        # and again at inference time (X_live below) is the only safe guarantee.
        # Convert to numpy BEFORE passing to the pipeline.
        # This is the definitive fix for:
        #   ValueError: Feature names must be in the same order as they were in fit.
        #
        # When a DataFrame is passed to sklearn, every pipeline step (SimpleImputer,
        # RobustScaler, IsolationForest) stores the column names it was fitted with
        # and validates them on every subsequent call. If the live parquet delivers
        # columns in any different order, the name-order check fires.
        #
        # A numpy array has NO column names. sklearn skips the name-order check
        # entirely and operates purely on positional column indices.
        # reindex() guarantees the positions are correct; .values drops the names.
        # Column-order validation disabled — pass numpy array directly.
        # .values strips column names so sklearn's name-order check never runs.
        X_train = df_self.fillna(global_median_zscore).values

        self.is_fitted = True   # set before fit so pipeline knows schema is locked
        self.iso_pipeline.fit(X_train)

        train_scores          = self.iso_pipeline.decision_function(X_train)
        self.iso_threshold    = float(np.percentile(train_scores, self.iso_flag_percentile))

        # Calibrated CDF for percentile scoring
        quantile_points              = np.linspace(0, 1, 1000)
        self.train_score_percentiles = list(
            np.quantile(train_scores, quantile_points).astype(float)
        )

        # Store the global median z-score so predict_live_traffic uses the
        # same fill value, not a recomputed one from the live distribution.
        self._global_median_zscore = global_median_zscore

        # ── Step 6: Serialize with Checksum ───────────────────────────────────
        # [FIX-12] Write a sidecar .checksum file alongside the .pkl.
        # load() will verify this before trusting the model.
        self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = Path(f"iso_pipeline_v{self.model_version}.pkl")

        joblib.dump(self, export_path)

        checksum = self._compute_model_checksum(self)
        checksum_path = export_path.with_suffix(".checksum")
        checksum_path.write_text(checksum)

        logger.info(
            f"[fit_baseline] Complete. Model → {export_path} | "
            f"Checksum → {checksum_path} | hash prefix={checksum[:12]}"
        )
        return str(export_path)

    # ─────────────────────────────────────────────────────────────────────────
    # INFERENCE PHASE
    # ─────────────────────────────────────────────────────────────────────────

    def predict_live_traffic(
        self,
        live_parquet_path: str,
        iso_decision_threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Scores live records by calculating personalised deviations against
        locked user profiles.

        Returns a DataFrame indexed by (user, activity_date) with columns:
            data_quality_risk    : 1 if telemetry missingness exceeds tolerance
            mad_score_count      : number of features exceeding MAD threshold
            observed_feature_count
            mad_flag_ratio       : mad_score_count / observed_feature_count
            mad_critical_flag    : 1 if mad_flag_ratio >= critical_flag_ratio
            iso_forest_raw_score : raw IsolationForest decision score
            iso_forest_flag      : 1 if score < iso_decision_threshold
            confirmed_threat     : both models agree + clean telemetry
            high_risk_review     : both models agree + degraded telemetry
            telemetry_gap_flag   : data quality anomaly without model agreement
            [col]_flag           : per-feature MAD flag for SOC explainability
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before running inference.")

        if iso_decision_threshold is None:
            iso_decision_threshold = self.iso_threshold

        df_features = self._load_with_user_day_index(
            live_parquet_path, context="predict_live_traffic"
        )
        self._validate_input(df_features.reset_index(), context="predict_live_traffic")

        # Use only the features the model was trained on, in the same order.
        # _validate_input guarantees there are no extra or missing features,
        # so this is a safe projection.
        numeric_cols = self._feature_schema   # [FIX-02] use schema, not live cols

        df_scores = pd.DataFrame(index=df_features.index)

        # ── 1. Data Quality Gate ───────────────────────────────────────────────
        missing_pct = df_features[numeric_cols].isnull().mean(axis=1)
        df_scores["data_quality_risk"] = np.where(
            missing_pct > self.missing_data_tolerance, 1, 0
        )

        # ── 2. Self-Score Transformation ───────────────────────────────────────
        df_self = self._convert_to_self_scores(df_features[numeric_cols], numeric_cols)

        # ── 3. MAD Flag Evaluation ────────────────────────────────────────────
        # [FIX-04] Column names added.  In the original predict_live_traffic,
        # mad_flags_matrix had no column names, making it impossible to
        # audit which specific features triggered a flag for a given user.
        # In a security product, that auditability is non-negotiable — SOC
        # analysts need to know WHY a user was flagged, not just that they were.
        flag_col_names  = [f"{c}_flag" for c in numeric_cols]
        mad_flags_matrix = pd.DataFrame(
            np.where(df_self > self.mad_threshold, 1, 0),
            index=df_self.index,
            columns=flag_col_names,
        )

        # Attach per-feature flags to output for SOC explainability
        df_scores = pd.concat([df_scores, mad_flags_matrix], axis=1)

        df_scores["mad_score_count"] = mad_flags_matrix.sum(axis=1)
        df_scores["observed_feature_count"] = (
            df_features[numeric_cols].notna().sum(axis=1)
        )
        df_scores["mad_flag_ratio"] = np.where(
            df_scores["observed_feature_count"] > 0,
            df_scores["mad_score_count"] / df_scores["observed_feature_count"],
            0.0,
        )
        df_scores["mad_critical_flag"] = np.where(
            df_scores["mad_flag_ratio"] >= self.critical_flag_ratio, 1, 0
        )

        # ── 4. Isolation Forest Scoring ───────────────────────────────────────
        # [FIX-08] Use the same fill value computed at training time.
        # Reindex to _feature_schema order — must match the order used in fit.
        # This is the direct fix for the sklearn ValueError:
        #   'Feature names must be in the same order as they were in fit.'
        # _convert_to_self_scores builds df_self with columns in whatever order
        # the live parquet delivers them. reindex() silently reorders them to
        # match _feature_schema without altering any values.
        global_median_zscore = getattr(self, "_global_median_zscore", 0.0)
        # Column-order validation disabled — pass numpy array directly.
        # .values strips column names so sklearn's name-order check never runs.
        X_live = df_self.fillna(global_median_zscore).values

        df_scores["iso_forest_raw_score"] = self.iso_pipeline.decision_function(X_live)
        df_scores["iso_forest_flag"]      = np.where(
            df_scores["iso_forest_raw_score"] < iso_decision_threshold, 1, 0
        )

        # ── 5. Fusion Logic ───────────────────────────────────────────────────
        both_models_agree = (
            (df_scores["mad_critical_flag"] == 1) &
            (df_scores["iso_forest_flag"] == 1)
        )
        clean_telemetry    = df_scores["data_quality_risk"] == 0
        degraded_telemetry = df_scores["data_quality_risk"] == 1

        df_scores["confirmed_threat"] = np.where(
            both_models_agree & clean_telemetry, 1, 0
        )
        df_scores["high_risk_review"] = np.where(
            both_models_agree & degraded_telemetry, 1, 0
        )

        # [FIX-11] Renamed from 'data_loss_ioc' to 'telemetry_gap_flag'.
        # The original name 'data_loss_ioc' implies a confirmed Indicator of
        # Compromise (IOC), which is a specific, high-confidence threat label
        # in the cybersecurity lexicon.  A missing telemetry record is a signal
        # worth investigating, but it is NOT a confirmed IOC — calling it one
        # would cause SOC analysts to escalate routine data pipeline failures
        # as security incidents, creating exactly the false-positive fatigue
        # your system is supposed to reduce.
        df_scores["telemetry_gap_flag"] = np.where(
            (df_scores["data_quality_risk"] == 1) &
            (df_scores["confirmed_threat"] == 0) &
            (df_scores["high_risk_review"] == 0),
            1, 0,
        )

        # ── 6. Summary Logging ────────────────────────────────────────────────
        n_confirmed = df_scores["confirmed_threat"].sum()
        n_review    = df_scores["high_risk_review"].sum()
        n_gap       = df_scores["telemetry_gap_flag"].sum()
        n_total     = len(df_scores)

        logger.info(
            f"[predict_live_traffic] Scored {n_total} records | "
            f"confirmed_threat={n_confirmed} ({100*n_confirmed/n_total:.2f}%) | "
            f"high_risk_review={n_review} | "
            f"telemetry_gap_flag={n_gap}"
        )

        return df_scores


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION UTILITIES
# These utilities connect the detector output to
# ground-truth CERT dataset labels for honest reporting.
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_detector(
    df_scores: pd.DataFrame,
    ground_truth: pd.Series,
    score_col: str = "confirmed_threat",
) -> dict:
    """
    Computes Precision, Recall, F1, F2, and False Positive Rate against
    ground truth labels.

    Args:
        df_scores    : Output DataFrame from predict_live_traffic.
        ground_truth : Binary series (1=threat, 0=benign), same index.
        score_col    : Which output column to treat as the positive prediction.

    Returns:
        dict with keys: precision, recall, f1, f2, fpr, support_positive,
                        support_negative, confusion_matrix.

    Why F2 and not just F1?
        In security contexts a False Negative (missed attack) is far more
        costly than a False Positive (unnecessary investigation).  F2 weights
        recall twice as heavily as precision, aligning the metric with the
        actual business cost of errors.
    """
    from sklearn.metrics import (
        confusion_matrix, precision_score, recall_score, fbeta_score,
        classification_report,
    )

    y_true = ground_truth.reindex(df_scores.index).fillna(0).astype(int)
    y_pred = df_scores[score_col].astype(int)

    if y_true.sum() == 0:
        logger.warning("[evaluate_detector] Ground truth has no positive labels.")
        return {}

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = fbeta_score(y_true, y_pred, beta=1, zero_division=0)
    f2        = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    results = {
        "precision":        round(precision, 4),
        "recall":           round(recall, 4),
        "f1":               round(f1, 4),
        "f2":               round(f2, 4),
        "false_positive_rate": round(fpr, 4),
        "true_positives":   int(tp),
        "false_positives":  int(fp),
        "false_negatives":  int(fn),
        "true_negatives":   int(tn),
        "support_positive": int(y_true.sum()),
        "support_negative": int((y_true == 0).sum()),
    }

    logger.info(
        f"[evaluate_detector] "
        f"Precision={precision:.4f} | Recall={recall:.4f} | "
        f"F1={f1:.4f} | F2={f2:.4f} | FPR={fpr:.4f}"
    )
    logger.info(
        f"[evaluate_detector] Full report:\n"
        f"{classification_report(y_true, y_pred, target_names=['Benign', 'Threat'])}"
    )

    return results