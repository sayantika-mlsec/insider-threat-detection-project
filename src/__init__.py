"""
Insider Threat Detection — production source modules.

Public API:
    from src import InsiderThreatDetector
    from src import extract_logon_features, extract_http_features, extract_email_features
    from src import fuse_feature_matrices
    from src import MASTER_FEATURE_SCHEMA, LOGON_FEATURES, HTTP_FEATURES, EMAIL_FEATURES
    from src import align_to_master_schema
"""

import logging

# Configure logging exactly once, at package import time.
# Modules below should NOT call basicConfig again — that would be a no-op
# and mask configuration drift. Use `logging.getLogger(__name__)` for
# module-level loggers if you want finer control later.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

from .schema import (
    LOGON_FEATURES,
    HTTP_FEATURES,
    EMAIL_FEATURES,
    MASTER_FEATURE_SCHEMA,
    align_to_master_schema,
)
from .extraction import (
    extract_logon_features,
    extract_http_features,
    extract_email_features,
)
from .fusion import fuse_feature_matrices
from .detector import InsiderThreatDetector

__all__ = [
    # schema
    "LOGON_FEATURES",
    "HTTP_FEATURES",
    "EMAIL_FEATURES",
    "MASTER_FEATURE_SCHEMA",
    "align_to_master_schema",
    # extraction
    "extract_logon_features",
    "extract_http_features",
    "extract_email_features",
    # fusion
    "fuse_feature_matrices",
    # detector
    "InsiderThreatDetector",
]