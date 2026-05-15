"""
Feature schema — the contract between extraction, fusion, and the detector.

Single source of truth for column names. If you add or rename a feature,
this is the only file you need to change; everything downstream picks it
up by reference.

new renames (vs. the pre-refactor schema):
    total_events           → total_event_buckets       (honest about SQL dedup)
    bytes_uploaded         → upload_event_count        (never was bytes)
    malicious_domain_hits  → keyword_match_indicator   (2-string heuristic)
"""

import numpy as np
import pandas as pd


LOGON_FEATURES = [
    'total_event_buckets',       # count of (user, pc, activity, 5-min-bucket) tuples
    'off_hours_ratio',
    'unique_systems_accessed',
    'failed_login_ratio',
    'active_orphan_sessions',
]

HTTP_FEATURES = [
    'url_visits',
    'upload_event_count',         # count of upload/post events, NOT bytes
    'keyword_match_indicator',    # 2-string heuristic; replace with URLhaus/OpenPhish
    'unique_external_domains',
    'after_hours_browsing',
]

EMAIL_FEATURES = [
    'emails_sent',
    'external_recipients',
    'after_hours_emails',
    'attachments_sent',
    'large_attachment_count',
]

MASTER_FEATURE_SCHEMA = LOGON_FEATURES + HTTP_FEATURES + EMAIL_FEATURES


def align_to_master_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures a DataFrame conforms to MASTER_FEATURE_SCHEMA.

    Missing columns are filled with NaN; column ordering is enforced.
    Returns a *copy* — never mutates the caller's DataFrame.
    """
    df = df.copy()
    for col in MASTER_FEATURE_SCHEMA:
        if col not in df.columns:
            df[col] = np.nan
    return df[MASTER_FEATURE_SCHEMA]