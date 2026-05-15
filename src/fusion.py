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

    # ── 4. IMPUTE: NaN → 0 only for known-active user-days ──────────────
    # If a user-day exists in the merged index, they did SOMETHING that
    # day. A NaN in a column means "this source had no events for that
    # user-day" — a legitimate zero. The detector's NaN audit treats
    # >20% NaN coverage as an IOC; fillna here means downstream zeros
    # are intentional, not silent data loss.
    df_fused = df_fused.fillna(0)

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