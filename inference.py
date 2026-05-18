"""
shap_explainer.py — SHAP Inference Layer for InsiderThreatDetector.

PURPOSE
───────
This module generates SHAP explanations for the Isolation Forest component
of InsiderThreatDetector. It is designed to answer ONE question that every
SOC analyst will ask the moment an alert fires:

    "WHY is this user flagged? Which specific behaviour drove this score?"

ARCHITECTURE DECISIONS (each one is explained where it's implemented)
──────────────────────────────────────────────────────────────────────
[ARCH-01] SHAP operates on the self-score space (z-scores), NOT raw features.
[ARCH-02] TreeExplainer is used, not KernelExplainer — speed is non-negotiable.
[ARCH-03] Background dataset is sampled from training self-scores, not live data.
[ARCH-04] Per-feature MAD flags from the detector are fused with SHAP values.
[ARCH-05] Explanations are generated only for flagged records (cost control).
[ARCH-06] Output is a structured dict ready for API serialisation (FastAPI/JSON).
[ARCH-07] Waterfall plots saved to disk with deterministic filenames.
[ARCH-08] SOC-readable plain-English reason strings generated per alert.
[ARCH-09] Global feature importance computed separately from instance explanations.
[ARCH-10] Explainer is cached — not rebuilt on every inference call.

CONTRACTS WITH detector.py
──────────────────────────────────────────
• Reads detector._feature_schema         → column order for SHAP input
• Reads detector._global_median_zscore   → NaN fill strategy (must match)
• Reads detector.iso_pipeline            → extracts the IsolationForest step
• Reads detector.iso_pipeline['scaler']  → re-applies same scaling to background
• Reads detector.baseline_stats          → per-user historical medians for context
• Reads detector.global_baseline_stats   → cold-start user context
• Reads detector.mad_threshold           → which features are MAD-flagged
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for servers with no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.detector import InsiderThreatDetector, MIN_HISTORY_DAYS

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("SHAPExplainer")


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FeatureContribution:
    """
    A single feature's contribution to a specific alert, in SOC-readable form.

    Why a dataclass instead of a raw dict?
    Because this object is serialised to JSON for the FastAPI alert endpoint.
    A typed dataclass gives you: IDE autocomplete, mypy validation, and a
    clear contract between this module and whatever consumes it.  Raw dicts
    silently accept typos in key names and break at runtime, not at write time.
    """
    feature_name:      str
    raw_value:         float          # The actual observed value (e.g. 47 USB transfers)
    user_baseline:     float          # User's personal historical median for this feature
    self_z_score:      float          # How many MAD units above their own baseline
    shap_value:        float          # SHAP contribution to the anomaly score
    mad_flagged:       bool           # Did this feature also exceed the MAD threshold?
    direction:         str            # "elevated" | "suppressed" | "normal"
    plain_english:     str            # Human-readable explanation for SOC analyst


@dataclass
class AlertExplanation:
    """
    Complete SHAP explanation for one (user, activity_date) alert record.

    This is what gets serialised into the alert payload sent to the SIEM.
    Every field has a specific consumer:
        user / activity_date    → alert routing
        threat_category         → SOC triage priority
        iso_raw_score           → raw signal for dashboards
        top_contributors        → analyst's primary focus
        all_contributions       → full audit trail
        narrative               → human-readable summary for L1 analysts
        plot_path               → link to waterfall chart in the alert UI
        baseline_source         → tells analyst whether to trust the baseline
    """
    user:               str
    activity_date:      str
    threat_category:    str           # "confirmed_threat" | "high_risk_review"
    iso_raw_score:      float
    mad_flag_ratio:     float
    top_contributors:   list[FeatureContribution]    # top-N by |shap_value|
    all_contributions:  list[FeatureContribution]    # full audit trail
    narrative:          str
    plot_path:          Optional[str] = None
    baseline_source:    str = "personal"  # "personal" | "global_cold_start"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXPLAINER CLASS
# ══════════════════════════════════════════════════════════════════════════════

class IsolationForestSHAPExplainer:
    """
    SHAP explanation layer for the Isolation Forest inside InsiderThreatDetector.

    Usage pattern:
        detector  = InsiderThreatDetector.load("model.pkl")
        explainer = IsolationForestSHAPExplainer(detector)
        explainer.fit_background(X_train_self_scores)

        df_scores = detector.predict_live_traffic("live.parquet")
        reports   = explainer.explain_flagged_records(
            df_scores, df_live_features, output_dir="shap_reports/"
        )
    """

    def __init__(
        self,
        detector: InsiderThreatDetector,
        top_n_features: int = 5,
        background_sample_size: int = 200,
        plot_output_dir: str = "shap_plots",
        generate_plots: bool = True,
    ):
        """
        Args:
            detector             : A fitted InsiderThreatDetector instance.
            top_n_features       : How many top SHAP features to surface per alert.
            background_sample_size: How many background samples for TreeExplainer.
            plot_output_dir      : Where to save waterfall plots.
            generate_plots       : Set False in unit tests or high-volume batch runs.
        """
        if not detector.is_fitted:
            raise RuntimeError(
                "[IsolationForestSHAPExplainer] Detector must be fitted before "
                "building an explainer."
            )

        self.detector              = detector
        self.top_n                 = top_n_features
        self.background_sample_size = background_sample_size
        self.plot_output_dir       = Path(plot_output_dir)
        self.generate_plots        = generate_plots

        # [ARCH-10] Explainer is built once and cached. Rebuilding shap.TreeExplainer
        # on every inference call costs ~200ms per call on a 200-estimator forest.
        # At 10,000 alerts/day, that's ~30 minutes of wasted CPU per day.
        self._explainer: Optional[shap.TreeExplainer] = None
        self._background: Optional[np.ndarray] = None
        self._is_explainer_ready: bool = False

        # [ARCH-02] Extract the IsolationForest step from the Pipeline.
        # We need the bare IsolationForest object, not the full pipeline, because
        # shap.TreeExplainer only accepts tree-based estimators directly.
        # The pipeline applies imputation + scaling before the forest — we must
        # manually replicate that transformation on any data we pass to SHAP.
        self._iso_forest = detector.iso_pipeline.named_steps["detector"]
        self._scaler = detector.iso_pipeline.named_steps["scaler"]
        self._imputer = detector.iso_pipeline.named_steps["imputer"]

        self.feature_names: list[str] = detector._feature_schema
        self.n_features = len(self.feature_names)

        logger.info(
            f"[IsolationForestSHAPExplainer] Initialised | "
            f"features={self.n_features} | top_n={self.top_n}"
        )

    def fit_background(self, X_train_self_scores: pd.DataFrame) -> None:
        """
        Builds the SHAP TreeExplainer using a background dataset.

        [ARCH-03] Why does background matter?
        SHAP TreeExplainer for IsolationForest computes the expected anomaly
        score E[f(x)] by integrating over a background distribution. The
        background defines what "normal" looks like to SHAP.

        WRONG approach: using live (flagged) data as background. That shifts
        the baseline toward anomalous behaviour and artificially deflates
        SHAP values — your explanations will understate how anomalous each
        feature is.

        CORRECT approach: sample from training self-scores (the same data the
        forest was trained on). This ensures SHAP's "expected value" is the
        expected score of a NORMAL user, making deviations meaningful.

        [ARCH-01] Input must be in self-score space (z-scores), NOT raw feature
        values. The IsolationForest was trained on self-scores. Passing raw
        values to SHAP would produce SHAP explanations for a completely
        different input space — the values would be meaningless.

        Args:
            X_train_self_scores : DataFrame of self-scores from fit_baseline
                                  (same format as df_self inside the detector).
        """
        if X_train_self_scores.shape[1] != self.n_features:
            raise ValueError(
                f"[fit_background] Expected {self.n_features} features, "
                f"got {X_train_self_scores.shape[1]}."
            )

        # [ARCH-03] Sample the background — we don't need all training rows.
        # 200 background samples is the standard SHAP recommendation for
        # TreeExplainer. Using more doesn't improve explanation quality but
        # linearly increases memory usage.
        n_available = len(X_train_self_scores)
        n_sample    = min(self.background_sample_size, n_available)

        fill_val = getattr(self.detector, "_global_median_zscore", 0.0)

        # Apply the same NaN fill strategy as the detector (FIX-08 in detector).
        # If we fill with a different value here, SHAP's background distribution
        # doesn't match what the model actually saw during training.
        background_raw = (
            X_train_self_scores
            .fillna(fill_val)
            .sample(n=n_sample, random_state=42)
            .values
        )

        # Re-apply the pipeline's imputer + scaler to the background.
        # We must pass pre-scaled data to shap.TreeExplainer when using
        # the raw IsolationForest step — the pipeline's transform() is not
        # applied automatically when we bypass it.
        background_transformed = self._imputer.transform(background_raw)
        background_transformed = self._scaler.transform(background_transformed)

        self._background = background_transformed

        logger.info(
            f"[fit_background] Background built | "
            f"n_samples={n_sample} (from {n_available} available) | "
            f"fill_value={fill_val:.4f}"
        )

        # [ARCH-02] TreeExplainer is correct for IsolationForest.
        # DO NOT use KernelExplainer here. KernelExplainer is model-agnostic
        # and uses a sampling approximation — it requires O(background_size × n_features)
        # forward passes per explained instance. For a 200-estimator forest with
        # 200 background samples, that's 40,000 model calls per alert.
        # TreeExplainer uses the tree structure directly: it's exact and runs
        # in O(n_estimators × max_depth) per instance — orders of magnitude faster.
        self._explainer = shap.TreeExplainer(
            self._iso_forest,
            data=self._background,
            feature_names=self.feature_names,
            # "tree_path_dependent" is the correct feature_perturbation for
            # unsupervised models like IsolationForest. "interventional" assumes
            # feature independence, which is wrong for correlated behavioural
            # features (e.g., login_count and active_hours are correlated).
            feature_perturbation="tree_path_dependent",
        )

        self._is_explainer_ready = True
        logger.info("[fit_background] TreeExplainer ready.")

    # ─────────────────────────────────────────────────────────────────────────
    # CORE PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _transform_for_shap(self, X_self_scores: np.ndarray) -> np.ndarray:
        """
        Apply imputer + scaler to self-score data before passing to SHAP.

        This replicates what iso_pipeline.decision_function() does internally,
        but exposes the intermediate representation so SHAP can operate on it.
        The TreeExplainer must receive the same input the IsolationForest trees
        actually see — post-imputation, post-scaling values.
        """
        X = self._imputer.transform(X_self_scores)
        X = self._scaler.transform(X)
        return X

    def _get_user_baseline(self, user: str) -> tuple[dict, str]:
        """
        Returns the user's baseline stats dict and a label indicating source.

        [ARCH-01 consequence] SHAP values are computed in self-score (z-score)
        space. To give the SOC analyst a human-readable explanation, we need to
        translate back to raw feature space: "This user normally transfers 2GB/day
        (their personal baseline). Yesterday they transferred 47GB (23 MAD units
        above their baseline)."

        This lookup is the bridge between SHAP's z-score world and the analyst's
        raw-value world.
        """
        n_days_in_baseline = len(
            self.detector.baseline_stats.get(user, {})
        )

        if (
            user not in self.detector.baseline_stats
            or n_days_in_baseline < MIN_HISTORY_DAYS
        ):
            return self.detector.global_baseline_stats, "global_cold_start"

        return self.detector.baseline_stats[user], "personal"

    def _build_feature_contribution(
        self,
        feature_name: str,
        shap_value: float,
        self_z_score: float,
        raw_value: float,
        user_baseline_stats: dict,
        mad_flagged: bool,
    ) -> FeatureContribution:
        """
        Builds a single FeatureContribution, including the plain-English
        reason string that an L1 SOC analyst reads in the alert UI.

        [ARCH-08] Why generate plain-English strings here?
        Because SHAP values are not human-readable. A SHAP value of -0.043 means
        nothing to an analyst triaging 200 alerts at 2am. The reason string
        converts the SHAP signal into actionable language: "data_transferred_gb
        was 47.3 GB — 23.1 standard deviations above this user's personal
        baseline of 2.1 GB. This was the #1 driver of the anomaly score."
        """
        col_stats      = user_baseline_stats.get(feature_name, {"median": 0.0, "mad": 1.0})
        user_median    = col_stats.get("median", 0.0)

        # Determine direction of deviation
        if self_z_score > self.detector.mad_threshold:
            direction = "elevated"
        elif self_z_score < 0:
            # Negative z-score means activity dropped significantly below baseline.
            # Suppression (e.g., a user who suddenly stops logging in) can be
            # just as anomalous as spikes.
            direction = "suppressed"
        else:
            direction = "normal"

        # [ARCH-08] Plain-English reason string
        if direction == "elevated":
            reason = (
                f"'{feature_name}' was {raw_value:.2f} — "
                f"{self_z_score:.1f} standard deviations above this user's "
                f"personal baseline of {user_median:.2f}."
            )
        elif direction == "suppressed":
            reason = (
                f"'{feature_name}' was {raw_value:.2f} — "
                f"significantly below this user's personal baseline of {user_median:.2f}. "
                f"Unusual suppression can indicate credential compromise or forced absence."
            )
        else:
            reason = (
                f"'{feature_name}' was {raw_value:.2f} "
                f"(within normal range; baseline: {user_median:.2f})."
            )

        if mad_flagged:
            reason += " ⚠ Also flagged by the MAD statistical model."

        return FeatureContribution(
            feature_name  = feature_name,
            raw_value     = round(float(raw_value), 4),
            user_baseline = round(float(user_median), 4),
            self_z_score  = round(float(self_z_score), 4),
            shap_value    = round(float(shap_value), 6),
            mad_flagged   = mad_flagged,
            direction     = direction,
            plain_english = reason,
        )

    def _generate_waterfall_plot(
        self,
        shap_values: np.ndarray,
        shap_base_value: float,
        feature_display_values: np.ndarray,
        user: str,
        activity_date: str,
        threat_category: str,
    ) -> str:
        """
        Saves a SHAP waterfall plot to disk and returns the file path.

        [ARCH-07] Why save to disk instead of returning bytes?
        Because in a production alert pipeline, the plot is consumed by
        two different systems on two different timescales:
            1. The SIEM embeds a link to the plot in the alert ticket (immediate).
            2. An analyst may pull the plot hours later during investigation.
        Returning bytes to the caller creates a memory burden for the orchestrator.
        A deterministic file path lets any downstream system retrieve it on demand.

        Filename format: shap_{user}_{date}_{threat_category}.png
        Deterministic: re-running for the same alert overwrites the old plot,
        preventing disk accumulation from repeated runs.
        """
        self.plot_output_dir.mkdir(parents=True, exist_ok=True)

        # Sanitise user ID for filesystem safety
        safe_user = str(user).replace("/", "_").replace("\\", "_").replace(" ", "_")
        safe_date = str(activity_date).replace(" ", "T").replace(":", "-")
        filename  = f"shap_{safe_user}_{safe_date}_{threat_category}.png"
        filepath  = self.plot_output_dir / filename

        shap_explanation = shap.Explanation(
            values       = shap_values,
            base_values  = shap_base_value,
            data         = feature_display_values,
            feature_names= self.feature_names,
        )

        fig, ax = plt.subplots(figsize=(10, max(4, self.n_features * 0.4)))

        shap.plots.waterfall(shap_explanation, show=False, max_display=self.top_n)

        plt.title(
            f"SHAP Waterfall — {user} | {activity_date} | [{threat_category.upper()}]",
            fontsize=11,
            pad=12,
        )
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)   # CRITICAL: always close — unclosed figures leak memory
                         # In a batch of 500 alerts, 500 unclosed figures will
                         # OOM-kill the process silently.

        logger.info(f"[_generate_waterfall_plot] Saved: {filepath}")
        return str(filepath)

    def _build_narrative(
        self,
        user: str,
        activity_date: str,
        threat_category: str,
        top_contributors: list[FeatureContribution],
        iso_raw_score: float,
        baseline_source: str,
    ) -> str:
        """
        [ARCH-08] Builds a 3-sentence narrative for L1 SOC analysts.

        The narrative is NOT a SHAP explanation. It's a plain-English summary
        that tells the analyst exactly what happened, why the model flagged it,
        and what to look at first — without requiring them to understand SHAP.

        Structure: WHO | WHAT | WHERE TO LOOK
        """
        top_feature = top_contributors[0] if top_contributors else None
        baseline_note = (
            "Note: This user has limited history — scored against company-wide baseline."
            if baseline_source == "global_cold_start"
            else ""
        )

        if top_feature:
            narrative = (
                f"User '{user}' on {activity_date} was flagged as '{threat_category}' "
                f"with an Isolation Forest anomaly score of {iso_raw_score:.4f} "
                f"(lower = more anomalous). "
                f"The strongest driver was '{top_feature.feature_name}': "
                f"{top_feature.plain_english} "
                f"Investigate the top {min(self.top_n, len(top_contributors))} "
                f"features listed below in priority order. {baseline_note}"
            )
        else:
            narrative = (
                f"User '{user}' on {activity_date} was flagged as '{threat_category}' "
                f"with anomaly score {iso_raw_score:.4f}. "
                f"No dominant feature driver identified. {baseline_note}"
            )

        return narrative.strip()

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC INTERFACE
    # ─────────────────────────────────────────────────────────────────────────

    def explain_single_record(
        self,
        user: str,
        activity_date: str,
        df_scores_row: pd.Series,
        df_features_row: pd.Series,
    ) -> AlertExplanation:
        """
        Generates a full SHAP explanation for a single (user, activity_date) alert.

        Args:
            user             : User identifier string.
            activity_date    : Date string of the flagged activity.
            df_scores_row    : One row from detector.predict_live_traffic() output.
            df_features_row  : One row of raw features from the live parquet.

        Returns:
            AlertExplanation dataclass, ready for JSON serialisation.
        """
        if not self._is_explainer_ready:
            raise RuntimeError(
                "[explain_single_record] Call fit_background() before explaining records."
            )

        feature_cols = self.detector._feature_schema
        fill_val     = getattr(self.detector, "_global_median_zscore", 0.0)

        # ── Step 1: Reconstruct self-scores for this single record ─────────────
        # We don't store df_self from predict_live_traffic because it's a
        # transient intermediate. We recompute it here for the single record.
        # This is correct because _convert_to_self_scores is a pure function
        # of (raw features, baseline_stats) — it always produces the same output.
        raw_values = df_features_row[feature_cols].copy()

        user_baseline_stats, baseline_source = self._get_user_baseline(user)

        self_z_scores = {}
        for col in feature_cols:
            col_stats   = user_baseline_stats.get(col, {"median": 0.0, "mad": 1.0})
            hist_median = col_stats.get("median", 0.0)
            hist_mad    = col_stats.get("mad", 1.0)
            raw_val     = raw_values.get(col, np.nan)

            if pd.isna(raw_val):
                self_z_scores[col] = fill_val
            elif hist_mad == 0:
                hist_p95 = col_stats.get("p95", hist_median)
                self_z_scores[col] = (
                    self.detector.mad_threshold + 1.0
                    if raw_val > hist_p95
                    else 0.0
                )
            else:
                self_z_scores[col] = abs(0.6745 * (raw_val - hist_median) / hist_mad)

        X_self = np.array([[self_z_scores[c] for c in feature_cols]])

        # ── Step 2: Transform for SHAP (impute + scale) ───────────────────────
        X_transformed = self._transform_for_shap(X_self)

        # ── Step 3: Compute SHAP values ───────────────────────────────────────
        # shap_vals shape: (1, n_features)
        # base_value: the expected model output over the background dataset.
        shap_result  = self._explainer(X_transformed)
        shap_vals    = shap_result.values[0]         # shape: (n_features,)
        base_value   = float(shap_result.base_values[0])

        # ── Step 4: Identify MAD-flagged features for this record ─────────────
        # [ARCH-04] Cross-referencing SHAP values with MAD flags gives the analyst
        # a dual-model confirmation: "SHAP says this feature is the top driver"
        # AND "the MAD model independently flagged the same feature."
        # That dual confirmation is far more credible than either signal alone.
        flag_col_names = {c: f"{c}_flag" for c in feature_cols}
        mad_flagged_set = {
            col for col in feature_cols
            if df_scores_row.get(flag_col_names[col], 0) == 1
        }

        # ── Step 5: Build FeatureContribution objects for all features ─────────
        all_contributions = []
        for i, col in enumerate(feature_cols):
            contrib = self._build_feature_contribution(
                feature_name       = col,
                shap_value         = shap_vals[i],
                self_z_score       = self_z_scores[col],
                raw_value          = float(raw_values.get(col, np.nan)),
                user_baseline_stats= user_baseline_stats,
                mad_flagged        = col in mad_flagged_set,
            )
            all_contributions.append(contrib)

        # Sort by |shap_value| descending — largest magnitude = most influential
        all_contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)
        top_contributors = all_contributions[: self.top_n]

        # ── Step 6: Determine threat category ─────────────────────────────────
        # Safely extract truthiness to handle bools, ints, or strings
        is_high_risk = df_scores_row.get("high_risk_review", 0) in [1, True, -1, "1", "True"]
        is_confirmed = df_scores_row.get("confirmed_threat", 0) in [1, True, -1, "1", "True"]
        is_telem_gap = df_scores_row.get("telemetry_gap_flag", 0) in [1, True, -1, "1", "True"]

        # Evaluate from MOST specific (3 flags) to LEAST specific (1 flag)
        if is_high_risk:
            # IF + MAD + DQ
            threat_category = "high_risk_review"
        elif is_confirmed:
            # IF + MAD only
            threat_category = "confirmed_threat"
        elif is_telem_gap:
            # DQ only
            threat_category = "telemetry_gap_flag"
        else:
            threat_category = "review"

        iso_raw_score = float(df_scores_row.get("iso_forest_raw_score", 0.0))
        mad_flag_ratio = float(df_scores_row.get("mad_flag_ratio", 0.0))

        # ── Step 7: Generate waterfall plot ───────────────────────────────────
        plot_path = None
        if self.generate_plots:
            # Pass the raw self-z-scores as display values (not the scaled
            # values). The analyst should read the axis in "z-score units",
            # not in the scaler's internal space.
            try:
                plot_path = self._generate_waterfall_plot(
                    shap_values           = shap_vals,
                    shap_base_value       = base_value,
                    feature_display_values= np.array([self_z_scores[c] for c in feature_cols]),
                    user                  = user,
                    activity_date         = str(activity_date),
                    threat_category       = threat_category,
                )
            except Exception as e:
                # Plot failure must NEVER suppress the explanation itself.
                # An alert without a chart is far better than a dropped alert.
                logger.warning(f"[explain_single_record] Plot generation failed: {e}")

        # ── Step 8: Build narrative ───────────────────────────────────────────
        narrative = self._build_narrative(
            user             = user,
            activity_date    = str(activity_date),
            threat_category  = threat_category,
            top_contributors = top_contributors,
            iso_raw_score    = iso_raw_score,
            baseline_source  = baseline_source,
        )

        return AlertExplanation(
            user             = str(user),
            activity_date    = str(activity_date),
            threat_category  = threat_category,
            iso_raw_score    = iso_raw_score,
            mad_flag_ratio   = mad_flag_ratio,
            top_contributors = top_contributors,
            all_contributions= all_contributions,
            narrative        = narrative,
            plot_path        = plot_path,
            baseline_source  = baseline_source,
        )

    def explain_flagged_records(
        self,
        df_scores: pd.DataFrame,
        df_live_features: pd.DataFrame,
        flag_col: str = "confirmed_threat",
        also_explain: Optional[list[str]] = None,
    ) -> list[AlertExplanation]:
        """
        Batch explains all flagged records from a predict_live_traffic() run.

        [ARCH-05] We only explain flagged records — not every scored record.
        On 100,000 records/day with 2% flagged, that's 2,000 SHAP calls vs
        100,000. Explaining benign records wastes CPU and produces noise in
        the alert queue.

        Args:
            df_scores        : Full output from predict_live_traffic().
            df_live_features : Raw feature DataFrame (same index as df_scores).
            flag_col         : Primary column to filter alerts by.
            also_explain     : Additional flag columns to include
                               (e.g., ["high_risk_review"]).

        Returns:
            List of AlertExplanation, one per flagged (user, activity_date).
        """
        if not self._is_explainer_ready:
            raise RuntimeError(
                "[explain_flagged_records] Call fit_background() first."
            )

        # Collect all rows where any of the requested flag columns is 1
        flag_cols = [flag_col] + (also_explain or [])
        available = [c for c in flag_cols if c in df_scores.columns]

        if not available:
            raise ValueError(
                f"[explain_flagged_records] None of the requested flag columns "
                f"found in df_scores: {flag_cols}. "
                f"Available columns: {list(df_scores.columns)}"
            )
        # Force a loud failure if a column is missing or misspelled
        missing = [c for c in flag_cols if c not in df_scores.columns]
        if missing:
            raise ValueError(f"Missing expected threat columns in live traffic: {missing}")
        
        # 2. Ensure the boolean mask handles specific types safely
        flagged_mask = df_scores[flag_cols].any(axis=1)
        flagged_index = df_scores[flagged_mask].index

        n_flagged = len(flagged_index)
        n_total   = len(df_scores)
        logger.info(
            f"[explain_flagged_records] Explaining {n_flagged} flagged records "
            f"out of {n_total} total ({100*n_flagged/n_total:.2f}%)."
        )

        explanations = []
        for i, idx in enumerate(flagged_index):
            user, activity_date = idx  # unpack MultiIndex tuple

            try:
                explanation = self.explain_single_record(
                    user             = user,
                    activity_date    = activity_date,
                    df_scores_row    = df_scores.loc[idx],
                    df_features_row  = df_live_features.loc[idx],
                )
                explanations.append(explanation)

                if (i + 1) % 50 == 0:
                    logger.info(f"[explain_flagged_records] Progress: {i+1}/{n_flagged}")

            except Exception as e:
                # Per-record failure must never halt the batch.
                # Log with full context so the on-call engineer knows exactly
                # which record failed and why, without losing all other alerts.
                logger.error(
                    f"[explain_flagged_records] Failed on ({user}, {activity_date}): {e}",
                    exc_info=True,
                )

        logger.info(
            f"[explain_flagged_records] Complete. "
            f"Successful explanations: {len(explanations)}/{n_flagged}"
        )
        return explanations

    def compute_global_feature_importance(
        self,
        X_train_self_scores: pd.DataFrame,
        sample_size: int = 500,
    ) -> pd.DataFrame:
        """
        Computes global SHAP feature importance across the training population.

        [ARCH-09] Global vs instance-level SHAP:
        Instance-level SHAP answers "why was THIS user flagged?"
        Global SHAP answers "which features drive anomaly detection for EVERYONE?"

        Global importance is used for:
            - Feature selection (dropping features with near-zero global SHAP)
            - Model auditing (if login_count dominates, is your model learning
              something meaningful or just proxying for role type?)
            - Communicating to stakeholders which behaviours the model monitors

        Returns:
            DataFrame with columns [feature, mean_abs_shap, rank]
            sorted by mean_abs_shap descending.
        """
        if not self._is_explainer_ready:
            raise RuntimeError(
                "[compute_global_feature_importance] Call fit_background() first."
            )

        fill_val   = getattr(self.detector, "_global_median_zscore", 0.0)
        n_sample   = min(sample_size, len(X_train_self_scores))

        X_sample = (
            X_train_self_scores
            .fillna(fill_val)
            .sample(n=n_sample, random_state=42)
            .values
        )
        X_transformed = self._transform_for_shap(X_sample)

        logger.info(
            f"[compute_global_feature_importance] Computing SHAP on "
            f"{n_sample} samples..."
        )

        shap_result = self._explainer(X_transformed)
        # shap_result.values shape: (n_samples, n_features)
        mean_abs_shap = np.abs(shap_result.values).mean(axis=0)

        importance_df = pd.DataFrame({
            "feature":        self.feature_names,
            "mean_abs_shap":  mean_abs_shap,
        })
        importance_df["rank"] = importance_df["mean_abs_shap"].rank(
            ascending=False, method="min"
        ).astype(int)
        importance_df = importance_df.sort_values("mean_abs_shap", ascending=False)

        logger.info(
            f"[compute_global_feature_importance] Done. "
            f"Top feature: '{importance_df.iloc[0]['feature']}' "
            f"(mean |SHAP|={importance_df.iloc[0]['mean_abs_shap']:.5f})"
        )

        return importance_df.reset_index(drop=True)

    def to_soc_report(self, explanations: list[AlertExplanation]) -> pd.DataFrame:
        """
        Flattens a list of AlertExplanation objects into a tabular SOC report.

        [ARCH-06] This is the final output layer — designed for two consumers:
            1. FastAPI endpoint: the calling function serialises this to JSON.
            2. Plotly dashboard: this DataFrame feeds the analyst's alert table.

        Each row is one alert. Top-N feature columns are expanded inline so
        the analyst can sort by any individual feature's contribution without
        opening individual waterfall plots.
        """
        rows = []
        for exp in explanations:
            row = {
                "user":            exp.user,
                "activity_date":   exp.activity_date,
                "threat_category": exp.threat_category,
                "iso_raw_score":   exp.iso_raw_score,
                "mad_flag_ratio":  exp.mad_flag_ratio,
                "baseline_source": exp.baseline_source,
                "narrative":       exp.narrative,
                "plot_path":       exp.plot_path,
            }
            # Expand top-N features into individual columns
            for rank, contrib in enumerate(exp.top_contributors, start=1):
                row[f"top{rank}_feature"]     = contrib.feature_name
                row[f"top{rank}_shap"]        = contrib.shap_value
                row[f"top{rank}_z_score"]     = contrib.self_z_score
                row[f"top{rank}_raw_value"]   = contrib.raw_value
                row[f"top{rank}_baseline"]    = contrib.user_baseline
                row[f"top{rank}_mad_flagged"] = contrib.mad_flagged
                row[f"top{rank}_direction"]   = contrib.direction

            rows.append(row)

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT — shows the complete usage pipeline end-to-end
# ══════════════════════════════════════════════════════════════════════════════

def run_shap_inference(
    model_path: str,
    train_parquet_path: str,
    live_parquet_path: str,
    output_dir: str = "shap_output",
    top_n: int = 5,
    generate_plots: bool = True,
) -> pd.DataFrame:
    """
    End-to-end SHAP inference pipeline.

    Designed to be called by:
        - A FastAPI endpoint (pass paths, receive the SOC report DataFrame)
        - A cron job (daily batch explanation run)
        - A Jupyter notebook (exploratory analysis)

    Args:
        model_path         : Path to the serialised InsiderThreatDetector .pkl
        train_parquet_path : Historical training data (for background sampling)
        live_parquet_path  : Live data to score and explain
        output_dir         : Root directory for SHAP plots and reports
        top_n              : Top N features to surface per alert
        generate_plots     : Whether to generate and save waterfall plots

    Returns:
        SOC report DataFrame — one row per flagged alert with SHAP columns.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Load fitted detector ────────────────────────────────────────────────
    logger.info(f"[run_shap_inference] Loading model from {model_path}...")
    detector = InsiderThreatDetector.load(model_path)

    # ── 2. Score live traffic ──────────────────────────────────────────────────
    logger.info("[run_shap_inference] Scoring live traffic...")
    df_live_features = InsiderThreatDetector._load_with_user_day_index(
        live_parquet_path, context="run_shap_inference"
    )
    df_scores = detector.predict_live_traffic(live_parquet_path)

    # ── 3. Build background from training data ─────────────────────────────────
    logger.info("[run_shap_inference] Building SHAP background from training data...")
    df_train = InsiderThreatDetector._load_with_user_day_index(
        train_parquet_path, context="run_shap_inference_background"
    )
    numeric_cols = detector._feature_schema
    fill_val     = getattr(detector, "_global_median_zscore", 0.0)

    # Recompute self-scores for the training set to use as background.
    # We cannot store df_self from fit_baseline (it would double memory usage
    # for the model artifact). Recomputing is cheap relative to retraining.
    df_train_self = detector._convert_to_self_scores(
        df_train[numeric_cols], numeric_cols
    )

    # ── 4. Initialise and fit explainer ───────────────────────────────────────
    explainer = IsolationForestSHAPExplainer(
        detector        = detector,
        top_n_features  = top_n,
        generate_plots  = generate_plots,
        plot_output_dir = str(output_path / "waterfall_plots"),
    )
    explainer.fit_background(df_train_self)

    # ── 5. Compute and log global feature importance ──────────────────────────
    importance_df = explainer.compute_global_feature_importance(df_train_self)
    importance_path = output_path / "global_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    logger.info(
        f"[run_shap_inference] Global importance saved to {importance_path}\n"
        f"{importance_df.to_string(index=False)}"
    )

    # ── 6. Explain all flagged records ────────────────────────────────────────
    explanations = explainer.explain_flagged_records(
        df_scores        = df_scores,
        df_live_features = df_live_features,
        flag_col         = "confirmed_threat",
        also_explain     = ["high_risk_review", "telemetry_gap_flag"],
    )

    # ── 7. Generate SOC report ────────────────────────────────────────────────
    soc_report = explainer.to_soc_report(explanations)

    # WHY PARQUET, NOT CSV:
    # The SOC report has typed columns — activity_date should be datetime,
    # confirmed_threat should be bool, iso_raw_score should be float32.
    # CSV serialises everything to strings and re-reads them as objects.
    # Streamlit then has to re-cast every column manually before filtering,
    # sorting, or plotting — and it silently gets it wrong on datetimes.
    # Parquet preserves dtypes exactly, compresses ~5x smaller than CSV,
    # and reads 10-50x faster with predicate pushdown for large alert tables.
    #
    # global_feature_importance stays as CSV: it's a 10-row human-readable
    # summary, not a queryable dataset. CSV is fine there.
    report_path = output_path / "soc_alert_report.parquet"
    soc_report.to_parquet(report_path, index=False, engine="pyarrow")
    logger.info(
        f"[run_shap_inference] SOC report saved to {report_path} | "
        f"Alerts: {len(soc_report)}"
    )

    # WHY A JSON SIDECAR:
    # The flat SOC report parquet holds one row per alert — good for the
    # summary table. But the Streamlit detail panel (when an analyst clicks
    # a specific alert) needs the full list[FeatureContribution] with
    # plain_english strings for ALL features, not just top-N.
    # Storing that nested structure in parquet requires complex list-of-struct
    # columns. JSON sidecar is simpler, still fast at this cardinality
    # (thousands of alerts, not millions), and directly loadable in Streamlit
    # with st.json() for debugging or json.load() for programmatic use.
    import json, dataclasses

    def _to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        raise TypeError(f"Cannot serialise {type(obj)}")

    sidecar_path = output_path / "soc_alert_detail.json"
    with open(sidecar_path, "w") as f:
        json.dump([dataclasses.asdict(e) for e in explanations], f, default=str)
    logger.info(f"[run_shap_inference] Detail sidecar saved to {sidecar_path}")

    return soc_report


# ══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT GUARD
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # WHY THIS BLOCK EXISTS:
    # Without this guard, running `python shap_explainer.py` directly would
    # silently do nothing — the module loads, defines classes and functions,
    # then exits. With this block, direct execution runs the full inference
    # pipeline using CLI arguments.
    #
    # WHY NOT at the top of the file:
    # argparse at module level runs on every import, including when
    # insider_threat_detector.py or dashboard.py imports this module.
    # That would crash any import with a "required argument missing" error.
    # Keeping it inside __main__ means imports are always safe.
    import argparse

    parser = argparse.ArgumentParser(
        description="Run SHAP inference on a fitted InsiderThreatDetector model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the serialised InsiderThreatDetector .pkl file.",
    )
    parser.add_argument(
        "--train",
        required=True,
        help="Path to the historical training parquet (for SHAP background sampling).",
    )
    parser.add_argument(
        "--live",
        required=True,
        help="Path to the live parquet file to score and explain.",
    )
    parser.add_argument(
        "--output",
        default="shap_output",
        help="Root output directory for parquet report, JSON sidecar, and plots.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        dest="top_n",
        help="Number of top SHAP features to surface per alert.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        dest="no_plots",
        help="Skip waterfall plot generation (faster for large batches).",
    )

    args = parser.parse_args()

    report = run_shap_inference(
        model_path         = args.model,
        train_parquet_path = args.train,
        live_parquet_path  = args.live,
        output_dir         = args.output,
        top_n              = args.top_n,
        generate_plots     = not args.no_plots,
    )

    print(f"\nDone. {len(report)} alerts written to {args.output}/")
    print(report[["user", "activity_date", "threat_category", "iso_raw_score"]].to_string(index=False))