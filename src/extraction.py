"""
DuckDB-based extraction functions for raw CERT r4.2 source files.

Each public function (`extract_logon_features`, `extract_http_features`,
`extract_email_features`) takes a CSV path and a date window, runs a DuckDB
query that emits a parquet, then aligns the output to MASTER_FEATURE_SCHEMA.

C4 fix retained: `start_date` and `end_date` are validated against
`^\\d{4}-\\d{2}-\\d{2}$` before being interpolated into the SQL string,
preventing date-parameter SQL injection.

Path sandboxing prevents directory traversal; only files under `./data/`
with `.csv` extension are accepted (unless `dev_mode=True`).
"""

import logging
import re
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from .schema import align_to_master_schema


# Date pattern shared across all three extraction functions
_DATE_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}$')


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers (DRY — these were duplicated ~30 lines each across functions)
# ─────────────────────────────────────────────────────────────────────────────
def _validate_extraction_inputs(
    user_input_path: str,
    start_date: str | None,
    end_date: str | None,
    dev_mode: bool,
) -> Path:
    """
    Shared input validation for all three extraction functions.

    Returns the resolved target file path on success.
    Raises ValueError / FileNotFoundError on any violation.
    """
    # ── PATH SANDBOX ────────────────────────────────────────────────────
    if dev_mode:
        logging.warning("Running in DEV_MODE. Strict security sandbox is DISABLED.")
        target_file: Path = Path(user_input_path).resolve()
        if not target_file.exists():
            raise FileNotFoundError(f"Cannot find file at {target_file}")
    else:
        safe_directory: Path = Path("data").resolve()
        target_file = (safe_directory / user_input_path).resolve()
        if not target_file.is_relative_to(safe_directory):
            raise ValueError(
                "Access Denied: Invalid file path "
                f"(directory traversal attempt: {user_input_path})"
            )

    if target_file.suffix != '.csv':
        raise ValueError(
            f"Access Denied: Only .csv files are permitted. Got {target_file.suffix}"
        )

    if "'" in target_file.as_posix():
        raise ValueError("File path contains illegal character (single quote).")

    # ── DATE PARAMETER VALIDATION (SQLi prevention — C4) ────────────────
    if start_date is not None and not _DATE_PATTERN.match(start_date):
        raise ValueError(
            f"start_date '{start_date}' is not YYYY-MM-DD. Example: '2010-01-01'"
        )
    if end_date is not None and not _DATE_PATTERN.match(end_date):
        raise ValueError(
            f"end_date '{end_date}' is not YYYY-MM-DD. Example: '2011-04-01'"
        )

    return target_file


def _build_output_path(output_prefix: str) -> Path:
    """Creates ./features/ if needed and returns a timestamped output path."""
    output_dir = Path("features").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{output_prefix}_{timestamp}.parquet"
    if "'" in output_file.as_posix():
        raise ValueError("Output path contains illegal character (single quote).")
    return output_file


def _build_date_filter_sql(start_date: str | None, end_date: str | None) -> str:
    """Builds the date filter SQL fragment used in all three extractions."""
    sql_fragment = ""
    if start_date:
        sql_fragment += (
            f" AND strptime(date, '%m/%d/%Y %H:%M:%S') "
            f">= TIMESTAMP '{start_date}'"
        )
    if end_date:
        sql_fragment += (
            f" AND strptime(date, '%m/%d/%Y %H:%M:%S') "
            f"< TIMESTAMP '{end_date}'"
        )
    return sql_fragment


# ─────────────────────────────────────────────────────────────────────────────
# Public extraction functions
# ─────────────────────────────────────────────────────────────────────────────
def extract_logon_features(
    user_input_path: str,
    start_date: str | None = None,
    end_date: str | None = None,
    output_prefix: str = "logon_features",
    dev_mode: bool = False,
) -> str:
    """
    Extracts daily behavioural features from raw logon logs.

    Output schema (after alignment to MASTER_FEATURE_SCHEMA):
        total_event_buckets       — count of (user, pc, activity, 5-min-bucket) tuples
        off_hours_ratio           — fraction of events outside 06:00–18:00 Mon–Fri
        unique_systems_accessed   — distinct PCs per day
        failed_login_ratio        — failed events / total event buckets
        active_orphan_sessions    — (logons − logoffs) per day
    """
    logging.info("Initiating Logon Feature Extraction Pipeline...")

    target_file = _validate_extraction_inputs(user_input_path, start_date, end_date, dev_mode)
    output_file = _build_output_path(output_prefix)
    date_filter_sql = _build_date_filter_sql(start_date, end_date)

    target_file_str = target_file.as_posix()
    output_file_str = output_file.as_posix()

    # NOTE on `total_event_buckets`:
    # The inner CTE GROUPs by (user, pc, activity, session_time), which
    # deduplicates events within the same 5-minute bucket. COUNT(*) outside
    # the CTE therefore counts BUCKETS, not raw events. Renamed from
    # `total_events`to reflect this honestly.
    query = f"""
    COPY (
        WITH SessionizedLogs AS (
            SELECT
                user,
                pc,
                activity,
                TIME_BUCKET(
                    INTERVAL '5 minutes',
                    strptime(date, '%m/%d/%Y %H:%M:%S')
                ) AS session_time
            FROM read_csv_auto('{target_file_str}')
            WHERE user != 'SYSTEM'
                AND (activity IN ('Logon', 'Logoff') OR activity ILIKE '%fail%')
                {date_filter_sql}
            GROUP BY user, pc, activity, session_time
        )
        SELECT
            user,
            CAST(session_time AS DATE) AS activity_date,

            COUNT(*) AS total_event_buckets,                  --honest name

            SUM(
                CASE
                    WHEN DAYOFWEEK(session_time) IN (0, 6) THEN 1
                    WHEN EXTRACT(hour FROM session_time) < 6
                      OR EXTRACT(hour FROM session_time) >= 18 THEN 1
                    ELSE 0
                END
            ) * 1.0 / NULLIF(COUNT(*), 0) AS off_hours_ratio,

            COUNT(DISTINCT pc) AS unique_systems_accessed,

            SUM(CASE WHEN activity ILIKE '%fail%' THEN 1 ELSE 0 END)
              * 1.0 / NULLIF(COUNT(*), 0) AS failed_login_ratio,

            SUM(CASE WHEN activity = 'Logon'  THEN 1 ELSE 0 END) -
            SUM(CASE WHEN activity = 'Logoff' THEN 1 ELSE 0 END) AS active_orphan_sessions

        FROM SessionizedLogs
        GROUP BY user, CAST(session_time AS DATE)
    ) TO '{output_file_str}' (FORMAT PARQUET);
    """

    try:
        duckdb.sql(query)
    except Exception as e:
        logging.error(f"[extract_logon_features] DuckDB execution failed. Reason: {e}")
        raise

    df_check = pd.read_parquet(output_file)
    if len(df_check) == 0:
        raise RuntimeError("Pipeline produced empty output. Check WHERE filters.")

    df_aligned = align_to_master_schema(df_check.set_index(['user', 'activity_date']))
    df_aligned.to_parquet(output_file, index=True)

    logging.info(
        f"Logon Pipeline Complete. | Rows: {len(df_aligned)} | "
        f"Schema: {list(df_aligned.columns)} | Output: {output_file}"
    )
    return str(output_file)


def extract_http_features(
    user_input_path: str,
    start_date: str | None = None,
    end_date: str | None = None,
    output_prefix: str = "http_features",
    dev_mode: bool = False,
) -> str:
    """
    Extracts daily HTTP behavioural features for web activity analysis.

    Output schema:
        url_visits               — total URL visits per day
        upload_event_count       — count of upload/post events (NOT bytes)
        keyword_match_indicator  — count of URLs matching 2 hardcoded strings 
                                   PRODUCTION: replace with URLhaus / OpenPhish
        unique_external_domains  — distinct URLs visited
        after_hours_browsing     — events outside 06:00–18:00
    """
    logging.info("Initiating HTTP Feature Extraction Pipeline...")

    target_file = _validate_extraction_inputs(user_input_path, start_date, end_date, dev_mode)
    output_file = _build_output_path(output_prefix)
    date_filter_sql = _build_date_filter_sql(start_date, end_date)

    target_file_str = target_file.as_posix()
    output_file_str = output_file.as_posix()

    # NOTE on `keyword_match_indicator`:
    # This is a TWO-STRING HEURISTIC for demo purposes only. It is NOT real
    # threat intelligence. Production replacement:
    #   1. Pull URLhaus or OpenPhish daily into a DuckDB table
    #   2. LEFT JOIN that table against http.url
    #   3. Count matches
    # Kept here so the rest of the fusion pipeline has a domain-reputation
    # signal during development; the column name is honest about the limitation.
    query = f"""
    COPY (
        WITH HttpLogs AS (
            SELECT
                user,
                url,
                activity,
                strptime(date, '%m/%d/%Y %H:%M:%S') AS raw_timestamp,
                CAST(strptime(date, '%m/%d/%Y %H:%M:%S') AS DATE) AS activity_date
            FROM read_csv('{target_file_str}', columns = {{
                'id':       'VARCHAR',
                'date':     'VARCHAR',
                'user':     'VARCHAR',
                'url':      'VARCHAR',
                'pc':       'VARCHAR',
                'activity': 'VARCHAR'
            }})
            WHERE user != 'SYSTEM' {date_filter_sql}
        )
        SELECT
            user,
            activity_date,

            COUNT(url) AS url_visits,

            SUM(CASE WHEN activity ILIKE '%upload%' OR activity ILIKE '%post%'
                     THEN 1 ELSE 0 END) AS upload_event_count,         

            SUM(CASE WHEN url ILIKE '%wikileaks%' OR url ILIKE '%keylog%'
                     THEN 1 ELSE 0 END) AS keyword_match_indicator,    

            COUNT(DISTINCT url) AS unique_external_domains,

            SUM(CASE WHEN EXTRACT(hour FROM raw_timestamp) < 6
                       OR EXTRACT(hour FROM raw_timestamp) >= 18
                     THEN 1 ELSE 0 END) AS after_hours_browsing

        FROM HttpLogs
        GROUP BY user, activity_date
    ) TO '{output_file_str}' (FORMAT PARQUET);
    """

    try:
        duckdb.sql(query)
    except Exception as e:
        logging.error(f"[extract_http_features] DuckDB execution failed. Reason: {e}")
        raise

    df_check = pd.read_parquet(output_file)
    if len(df_check) == 0:
        raise RuntimeError("Pipeline produced empty output. Check WHERE filters.")

    df_aligned = align_to_master_schema(df_check.set_index(['user', 'activity_date']))
    df_aligned.to_parquet(output_file, index=True)

    logging.info(f"HTTP Pipeline Complete. | Rows: {len(df_aligned)} | Output: {output_file}")
    return str(output_file)


def extract_email_features(
    user_input_path: str,
    start_date: str | None = None,
    end_date: str | None = None,
    output_prefix: str = "email_features",
    dev_mode: bool = False,
) -> str:
    """Extracts daily Email behavioural features for communication analysis."""
    logging.info("Initiating Email Feature Extraction Pipeline...")

    target_file = _validate_extraction_inputs(user_input_path, start_date, end_date, dev_mode)
    output_file = _build_output_path(output_prefix)
    date_filter_sql = _build_date_filter_sql(start_date, end_date)

    target_file_str = target_file.as_posix()
    output_file_str = output_file.as_posix()

    query = f"""
    COPY (
        WITH EmailLogs AS (
            SELECT
                user,
                "to" AS recipient,
                attachment_count,
                size AS email_size,
                strptime(date, '%m/%d/%Y %H:%M:%S') AS raw_timestamp,
                CAST(strptime(date, '%m/%d/%Y %H:%M:%S') AS DATE) AS activity_date
            FROM read_csv('{target_file_str}', columns = {{
                'id':               'VARCHAR',
                'date':             'VARCHAR',
                'user':             'VARCHAR',
                'pc':               'VARCHAR',
                'to':               'VARCHAR',
                'cc':               'VARCHAR',
                'bcc':              'VARCHAR',
                'from':             'VARCHAR',
                'size':             'VARCHAR',
                'attachment_count': 'VARCHAR',
                'content':          'VARCHAR'
            }})
            WHERE user != 'SYSTEM' {date_filter_sql}
        )
        SELECT
            user,
            activity_date,

            COUNT(*) AS emails_sent,

            -- External = recipient NOT in internal domain (CERT uses 'dtaa.com')
            SUM(CASE WHEN recipient NOT ILIKE '%@dtaa.com%' THEN 1 ELSE 0 END) AS external_recipients,

            SUM(CASE WHEN EXTRACT(hour FROM raw_timestamp) < 6
                       OR EXTRACT(hour FROM raw_timestamp) >= 18
                     THEN 1 ELSE 0 END) AS after_hours_emails,

            -- Attachments are semicolon-separated; count = N(semicolons) + 1
            SUM(
                CASE
                    WHEN attachment_count IS NOT NULL AND attachment_count != ''
                    THEN LENGTH(attachment_count) - LENGTH(REPLACE(attachment_count, ';', '')) + 1
                    ELSE 0
                END
            ) AS attachments_sent,

            -- > 1MB threshold for "large" attachment
            SUM(CASE WHEN CAST(email_size AS INTEGER) > 1000000 THEN 1 ELSE 0 END) AS large_attachment_count

        FROM EmailLogs
        GROUP BY user, activity_date
    ) TO '{output_file_str}' (FORMAT PARQUET);
    """

    try:
        duckdb.sql(query)
    except Exception as e:
        logging.error(f"[extract_email_features] DuckDB execution failed. Reason: {e}")
        raise

    df_check = pd.read_parquet(output_file)
    if len(df_check) == 0:
        raise RuntimeError("Pipeline produced empty output. Check WHERE filters.")

    df_aligned = align_to_master_schema(df_check.set_index(['user', 'activity_date']))
    df_aligned.to_parquet(output_file, index=True)

    logging.info(f"Email Pipeline Complete. | Rows: {len(df_aligned)} | Output: {output_file}")
    return str(output_file)