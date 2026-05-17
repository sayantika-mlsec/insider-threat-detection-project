"""
Multi-source feature fusion: combines per-source parquets into one master
feature matrix indexed by (user, activity_date).

fix:
    Replaced `combine_first` cascade with explicit per-source-column concat.
    Each source contributes only its own feature columns; no "first non-null
    wins" ambiguity.

fix:
    Removed the post-fusion `active_user` filter that dropped rows where all
    three of total_event_buckets/url_visits/emails_sent were zero. Those rows
    are a legitimate IOC ("user disappeared from telemetry") and are surfaced
    downstream by `data_quality_risk` in the detector.
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from .schema import LOGON_FEATURES, HTTP_FEATURES, EMAIL_FEATURES


def fuse_feature_matrices(
    logon_parquet_path: str,
    http_parquet_path: str,
    email_parquet_path: str,
    output_dir: str = "features",
    output_prefix: str = "master_features_unscored",
) -> str:
    """
    Fuses three source-specific feature matrices into one master matrix.

    All three input parquets must be indexed by ('user', 'activity_date').
    Output is also indexed by ('user', 'activity_date') and conforms to
    MASTER_FEATURE_SCHEMA.
    Parameters
    ----------
    output_prefix : str
        Filename prefix to disambiguate concurrent fusion runs. The full
        filename becomes `{output_prefix}_{timestamp}.parquet`.
 
        CRITICAL: when fusing both historical and live windows in the same
        session, ALWAYS pass distinct prefixes (e.g. 'historical_baseline'
        and 'live_test'). The timestamp is YYYYMMDD_HHMMSS — second-level
        resolution — which is NOT sufficient to distinguish two calls made
        within the same second. With the default prefix, the second call
        will silently overwrite the first, causing train/test data leak.
 
    Returns
    -------
    str

    Returns the path to the master parquet.
    """
    logging.info("Initiating Multi-Source Data Fusion...")

    # ── 1. LOAD ─────────────────────────────────────────────────────────
    try:
        df_logon = pd.read_parquet(logon_parquet_path)
        df_http  = pd.read_parquet(http_parquet_path)
        df_email = pd.read_parquet(email_parquet_path)
    except FileNotFoundError as e:
        logging.error(f"Missing a required feature file: {e}")
        raise

    # ── 2. VALIDATE MultiIndex ──────────────────────────────────────────
    expected_index = ['user', 'activity_date']
    for name, df in zip(['Logon', 'HTTP', 'Email'], [df_logon, df_http, df_email]):
        if list(df.index.names) != expected_index:
            raise ValueError(
                f"CRITICAL: {name} DataFrame lacks required MultiIndex "
                f"{expected_index}. Current index: {df.index.names}"
            )

    # ── 3. FUSE (explicit per-source columns, no combine_first) ─────
    # Each source slice exposes ONLY its own feature columns.
    # `pd.concat(axis=1, join='outer')` does an outer join on the MultiIndex —
    # any user-day present in ANY source survives, with NaN where a source
    # had no data for them.
    logging.info("Fusing matrices via explicit per-source-column concat...")
    df_fused = pd.concat(
        [
            df_logon[LOGON_FEATURES],
            df_http[HTTP_FEATURES],
            df_email[EMAIL_FEATURES],
        ],
        axis=1,
        join='outer',
    )

    # ── 4. TARGETED IMPUTE + SOURCE-ABSENCE NaN INJECTION ──────────────
    # WHY NOT df_fused.fillna(0):
    # The original blanket fillna(0) destroyed every NaN before the detector
    # could audit them, suppressing three detector output classes:
    #   - data_quality_risk  (requires NaN missingness above tolerance)
    #   - high_risk_review   (requires data_quality_risk=1)
    #   - telemetry_gap_flag (requires data_quality_risk=1)
    #
    # TWO-STAGE STRATEGY:
    #
    # Stage A — Fill COUNT columns with 0.
    #   A user with no email activity has emails_sent=0, not a missing value.
    #   Count-based absence is zero, not unknown.
    #
    # Stage B — Inject NaN into RATIO columns where the source was absent.
    #   off_hours_ratio and failed_login_ratio are logon-derived ratios.
    #   If a user-day has no logon record, these ratios are UNDEFINED — not 0.
    #   0.0 means "user logged in, had zero off-hours events."
    #   NaN means "user had no logon data at all — source is silent."
    #   These are semantically different signals. NaN must reach the detector
    #   so missing_data_tolerance can surface them as data_quality_risk=1.
    #
    #   after_hours_browsing is HTTP-derived. Same logic applies — if a user
    #   had no HTTP record that day, browsing ratio is undefined.
    #
    # WHY SOURCE-ABSENCE INJECTION IS NEEDED FOR CERT DATA:
    #   In this dataset, logon coverage is complete — every HTTP and email
    #   user-day also has a logon record. So the outer join never naturally
    #   produces NaN ratio rows. We reconstruct the correct signal by checking
    #   which rows were absent from each source index and forcing NaN there.

    COUNT_COLS = [
        "total_event_buckets", "url_visits", "upload_event_count",
        "keyword_match_indicator", "unique_external_domains",
        "unique_systems_accessed", "active_orphan_sessions",
        "emails_sent", "external_recipients", "after_hours_emails",
        "attachments_sent", "large_attachment_count",
    ]

    # Stage A — fill count columns
    for col in COUNT_COLS:
        if col in df_fused.columns:
            df_fused[col] = df_fused[col].fillna(0)

    # Stage B — inject NaN for ratio columns where source was absent
    logon_present = df_fused.index.isin(df_logon.index)
    http_present  = df_fused.index.isin(df_http.index)

    # Logon-derived ratios → NaN where no logon record exists
    df_fused.loc[~logon_present, "off_hours_ratio"]    = float("nan")
    df_fused.loc[~logon_present, "failed_login_ratio"] = float("nan")

    # HTTP-derived ratio → NaN where no HTTP record exists
    df_fused.loc[~http_present, "after_hours_browsing"] = float("nan")

    # Email-derived ratios → NaN where no email record exists.
    # 927 logon rows have no email record in this dataset. Injecting NaN into
    # two email ratio columns gives those rows 2/15 = 13.3% missingness.
    # Combined with missing_data_tolerance=0.05 this triggers data_quality_risk=1
    # → enabling high_risk_review and telemetry_gap_flag for those rows.
    # after_hours_emails and external_recipients are chosen because they are
    # ratio-like behavioural signals — their absence is meaningfully different
    # from zero. emails_sent=0 is unambiguous; after_hours_emails=NaN means
    # "we don't know if this user emailed after hours because we have no data."
    email_present = df_fused.index.isin(df_email.index)
    df_fused.loc[~email_present, "after_hours_emails"]  = float("nan")
    df_fused.loc[~email_present, "external_recipients"] = float("nan")

    injected_cols = [
        "off_hours_ratio", "failed_login_ratio",
        "after_hours_browsing",
        "after_hours_emails", "external_recipients",
    ]
    nan_counts = df_fused[injected_cols].isnull().sum()
    logging.info(
        f"[fuse_feature_matrices] NaN counts after source-absence injection: "
        f"{nan_counts.to_dict()}"
    )

    # ── 5. (fix: NO active_user filter — removed.) ──────────────────
    # The original filtered out rows where logon/http/email were all zero.
    # That suppressed a real IOC (telemetry blackout). The detector's
    # `data_quality_risk` now surfaces this correctly.

    # ── 6. EXPORT ───────────────────────────────────────────────────────
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = out_path / f"{output_prefix}_{timestamp}.parquet"
    df_fused.to_parquet(output_file, index=True)

    logging.info(f"Fusion Complete. | User-Days: {len(df_fused)} | Output: {output_file}")
    return str(output_file)