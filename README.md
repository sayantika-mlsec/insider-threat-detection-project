# Insider Threat Detection System
### Dual-Model Behavioral Anomaly Detection for Enterprise Security

A production-oriented ML pipeline that detects anomalous user behavior in enterprise environments using a **Self-as-Baseline architecture** — comparing each user only against their own historical profile, not the global population. Built on the CERT Insider Threat Dataset r4.2, with a dual-model fusion layer, SHAP-powered SOC analyst explainability, and a full diagnostic suite.

> **Designed for:** SOC analysts and security engineers who need to triage insider threat alerts with explainable, per-user behavioral context — not just a flag that says "anomalous."

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Why This Is Hard](#why-this-is-hard)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Dual-Model Approach](#dual-model-approach)
- [Fusion Logic](#fusion-logic)
- [Explainability Layer](#explainability-layer)
- [Results](#results)
- [Dashboard](#dashboard)
- [Repo Structure](#repo-structure)
- [Quickstart](#quickstart)
- [Known Limitations & Production Gaps](#known-limitations--production-gaps)
- [Future Work](#future-work)

---

## Problem Statement

Insider threats — malicious or negligent actions by employees with legitimate access — cause an estimated **$15.4M per incident** in enterprise environments (Ponemon Institute, 2022). Unlike external attacks, insiders already have valid credentials, making signature-based detection useless.

The SOC analyst's problem is not detection sensitivity — it's **alert fatigue**. Existing tools generate thousands of weekly flags, the vast majority false positives. Analysts spend more time dismissing noise than investigating real threats.

This project builds a detection system that answers three questions simultaneously:
1. **Is this user behaving anomalously?** (Detection)
2. **How anomalous, relative to their own baseline?** (Severity scoring)
3. **Which specific behaviors drove the flag — and why?** (Explainability for triage)

---

## Why This Is Hard

**Extreme class imbalance.** In the CERT r4.2 dataset, malicious users represent roughly 2% of the population. Standard accuracy metrics are meaningless — a model that flags nobody scores 98% accuracy. This project optimizes for **F2-score**, which weights recall twice as heavily as precision, reflecting the real cost asymmetry: a missed insider threat is far more damaging than a false positive investigation.

**The Global Baseline Trap.** Most anomaly detection systems compare each user to the population mean. A sysadmin transferring 50GB/day looks like an extreme outlier against a population of office workers — even though that's completely normal for their role. This system avoids the trap by comparing **each user only to their own historical profile** (Self-as-Baseline).

**Cold start problem.** New employees have no personal baseline. Routing them directly to per-user scoring would flag every legitimate action on day one. This system routes users with fewer than 5 days of history to a global population baseline, preventing false positives on new hires.

**Label scarcity.** Ground truth insider threat labels are rare and delayed — real organizations often don't know a threat occurred until months later. This system's primary detection path is **fully unsupervised**, requiring no labels at training time.

---

## Architecture

```
Raw CERT Logs (logon / http / email CSVs)
        │
        ▼
┌───────────────────────────────────┐
│   DuckDB Feature Extraction       │  ← Per-source behavioral features
│   (extraction.py)                 │    SQL-based, parquet output
└───────────────┬───────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│   Multi-Source Fusion Layer       │  ← Outer join on (user, activity_date)
│   (fusion.py)                     │    NaN semantics preserved per source
└───────────────┬───────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────┐
│   InsiderThreatDetector  (detector.py)                    │
│                                                           │
│   Self-as-Baseline Transform                              │
│   (raw features → per-user robust z-scores)               │
│                    │                                      │
│         ┌──────────┴──────────┐                           │
│         ▼                     ▼                           │
│   MAD Statistical         Isolation Forest                │
│   Baseline Model          (sklearn Pipeline:              │
│   (robust z-score         SimpleImputer →                 │
│    threshold)             RobustScaler →                  │
│                           IsolationForest)                │
│         │                     │                           │
│         └──────────┬──────────┘                           │
│                    ▼                                      │
│             Fusion Logic                                  │
│    confirmed_threat / high_risk_review /                  │
│    telemetry_gap_flag                                     │
└───────────────┬───────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│   SHAP Explainability Layer       │  ← Per-alert waterfall plots
│   (inference.py)                  │    Plain-English reason strings
└───────────────┬───────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│   Plotly Dashboard                │  ← SOC analyst triage interface
│   (dashboard.py)                  │    Risk scores, anomaly heatmaps
└───────────────────────────────────┘
```

---

## Dataset

**CERT Insider Threat Dataset r4.2** — Carnegie Mellon University SEI  
Registration required at: [sei.cmu.edu/data-access](https://www.sei.cmu.edu/data-access)

| Property | Value |
|---|---|
| Users | ~1,000 synthetic employees |
| Timespan | 18 months of simulated enterprise activity |
| Sources | Logon, HTTP browsing, Email, File, Device |
| Sources Used | Logon, HTTP, Email (3 of 5) |
| Malicious Users | ~2% of population |
| Label Availability | Ground truth labels available (used for evaluation only, not training) |

**Why only 3 of 5 sources?** File and Device logs require additional preprocessing for byte-level features (file size normalization, USB transfer parsing) that are out of scope for this phase. The three sources used cover the behavioral signals most commonly available in real SIEM environments.

---

## Feature Engineering

15 behavioral features engineered across three log sources. Every feature was chosen because it maps to a known MITRE ATT&CK technique or established insider threat indicator.

### Logon Features

| Feature | Description | Threat Signal |
|---|---|---|
| `total_event_buckets` | Count of (user, pc, activity, 5-min) tuples | Unusual session volume |
| `off_hours_ratio` | % of logins outside Mon–Fri 06:00–18:00 | T1078 — Valid Accounts abuse after hours |
| `unique_systems_accessed` | Distinct PCs per day | Lateral movement indicator |
| `failed_login_ratio` | Failed events / total event buckets | T1110 — Brute Force / credential testing |
| `active_orphan_sessions` | Logons minus Logoffs per day | Unclosed sessions / shared credential use |

### HTTP Features

| Feature | Description | Threat Signal |
|---|---|---|
| `url_visits` | Total URL visits per day | Browsing volume spike |
| `upload_event_count` | Count of upload/POST events | T1048 — Exfiltration over web |
| `keyword_match_indicator` | URL matches against known-bad patterns | Domain reputation signal ⚠ |
| `unique_external_domains` | Distinct external URLs visited | C2 communication breadth |
| `after_hours_browsing` | Browsing events outside business hours | Covert exfiltration window |

> ⚠ `keyword_match_indicator` is a two-string heuristic in this version. See [Known Limitations](#known-limitations--production-gaps).

### Email Features

| Feature | Description | Threat Signal |
|---|---|---|
| `emails_sent` | Total emails sent per day | Volume-based exfiltration signal |
| `external_recipients` | Emails to non-company domains | T1048 — Data exfiltration via email |
| `after_hours_emails` | Emails sent outside business hours | Covert communication |
| `attachments_sent` | Total attachment count | File staging before exfiltration |
| `large_attachment_count` | Attachments > 1MB | Bulk data exfiltration indicator |

---

## Dual-Model Approach

Two independent models score each user-day. Neither model alone is the decision — only agreement between both triggers a confirmed alert.

### Model 1 — MAD Statistical Baseline (Per-User Robust Z-Score)

Each user's historical median and MAD (Median Absolute Deviation) is computed per feature. At inference, each feature value is converted to a robust z-score relative to that user's own history:

```
z = |0.6745 × (live_value − user_median) / user_MAD|
```

A user-day is flagged if the ratio of anomalous features exceeds a data-driven `critical_flag_ratio` threshold (derived from the 90th percentile of training flag distributions).

**Why MAD instead of standard deviation?** Standard deviation is sensitive to the outliers it's trying to detect — a single extreme event inflates the standard deviation and suppresses future detections. MAD is robust to exactly this failure mode.

### Model 2 — Isolation Forest (Unsupervised Ensemble)

Trained on the full Self-as-Baseline feature matrix (all users' z-scores combined). The forest learns which combinations of behavioral deviations are globally unusual — not just individually high-scoring features.

Key configuration decisions:
- `contamination=0.02` — reflects the actual ~2% malicious user prevalence in CERT r4.2. sklearn's default of `0.1` would force 10% of training users into the anomaly class, producing a wildly miscalibrated boundary.
- `n_estimators=200` — increased from sklearn's default 100 for stability on sparse behavioral data.
- `RobustScaler` — used instead of StandardScaler because behavioral features have heavy-tailed distributions (a single exfiltration event produces extreme values that would collapse StandardScaler's normalization).

**Why unsupervised?** Ground truth labels are unavailable at training time in most real deployments. The model must detect threats it has never seen labeled examples of.

### Model Comparison

| Property | MAD Baseline | Isolation Forest |
|---|---|---|
| Supervision required | None | None |
| Interpretability | High — per-feature z-scores | Low — ensemble paths |
| Handles correlated features | No | Yes |
| Robust to new attack patterns | Moderate | High |
| Cold start behavior | Falls back to global median | Falls back to global median |
| False positive rate | Higher (single-feature sensitivity) | Lower (multi-feature combinations) |

---

## Fusion Logic

Both models must agree before a confirmed alert is raised. This dual-confirmation requirement is the primary false-positive reduction mechanism.

```
confirmed_threat  = MAD_flag AND ISO_flag AND clean_telemetry
high_risk_review  = MAD_flag AND ISO_flag AND degraded_telemetry
telemetry_gap_flag = data_quality_risk AND NOT confirmed AND NOT review
```

**Why a fusion layer instead of just using the better model?**  
The two models catch different threat patterns. MAD catches single-feature spikes (e.g., a user who suddenly sends 10× their usual email volume). Isolation Forest catches multi-feature combinations (e.g., a user whose email volume, off-hours ratio, and external recipient count all shift slightly together — no single feature is extreme, but the combination is unusual). Requiring agreement filters out the noise each model generates independently.

---

## Explainability Layer

Every confirmed alert is accompanied by a SHAP waterfall explanation generated by `IsolationForestSHAPExplainer`.

**Why SHAP for IsolationForest?** An alert that says "this user is anomalous" is useless to a SOC analyst. An alert that says "this user's `external_recipients` was 4.2 standard deviations above their personal baseline of 2 emails/day, and `after_hours_emails` was simultaneously elevated — both independently flagged by the MAD model" is actionable.

Each alert produces:
- A **SHAP waterfall plot** showing the top-5 feature contributions to the anomaly score
- A **plain-English narrative** string written for L1 analysts with no ML background
- A **dual-confirmation indicator** showing whether each SHAP-identified feature was also independently flagged by the MAD model
- A **baseline source label** — "personal baseline" vs "global cold-start" — so analysts know how much to trust the score

**Architecture note:** SHAP operates on the self-score (z-score) space, not raw feature values. The `TreeExplainer` with `feature_perturbation="tree_path_dependent"` is used — not `KernelExplainer` — because behavioral features are correlated and `KernelExplainer`'s independence assumption would produce misleading attributions.

---

## Results

Precision/Recall/F2 evaluation requires the CERT r4.2 scenario answer key, which CMU SEI withholds from the dataset download and releases on request. The `evaluate_detector()` function in `detector.py` is fully implemented and ready to run once labels are obtained. This is a known limitation of working with the CERT dataset — in real enterprise deployments, ground truth insider threat labels are equally rare and delayed, making unsupervised evaluation the practical standard.

> **Dataset note:** CERT r4.2 is explicitly described by CMU as a "dense needles" dataset — the volume of synthetic malicious activity is unrealistically high compared to real enterprise environments. Model performance on this dataset is therefore an optimistic upper bound, not a production baseline.

**Diagnostic output — 84,029 scored records:**

| Metric | Value | Notes |
|---|---|---|
| Total records scored | 84,029 | Full live window |
| Users scored | 889 / 1,000 trained | 100% with personal baseline — zero cold-start |
| confirmed_threat flags | 280 (0.33%) | Both models agree + clean telemetry |
| high_risk_review flags | 120 (0.14%) | Both models agree + degraded telemetry |
| telemetry_gap_flag | 807 (0.96%) | Data quality anomaly, models did not agree |
| Both models agreed | 400 records | Fusion layer input |
| ISO Forest flagged | 496 records (0.59%) | Below iso_threshold |
| MAD critical flag | 2,196 records (2.61%) | Above critical_flag_ratio |
| Distribution drift | −0.0046 ✓ | Within acceptable range — no recalibration needed |
| Top flagged user | HVF0067 | 8 confirmed_threat days |

**Top 10 users by confirmed_threat days:**

| User | Threat Days |
|---|---|
| HVF0067 | 8 |
| IRM0931 | 7 |
| PPF0435 | 6 |
| GAF0570 | 6 |
| NSC0597 | 6 |
| PTH0552 | 6 |
| MAR0955 | 5 |
| MBJ0312 | 5 |
| KRM0541 | 5 |
| SBH0537 | 5 |

**Top features by MAD flag rate:**

| Feature | Flag Rate | Signal |
|---|---|---|
| attachments_sent | 11.84% | Primary exfiltration signal |
| emails_sent | 11.84% | Volume-based exfiltration |
| active_orphan_sessions | 6.45% | Unclosed sessions / shared credentials |
| off_hours_ratio | 4.42% | After-hours activity |
| total_event_buckets | 2.76% | Unusual session volume |
| failed_login_ratio | 0.00% | ⚠ Dead feature in this dataset |
| keyword_match_indicator | 0.00% | ⚠ Dead feature — heuristic limitation |
| large_attachment_count | 0.00% | ⚠ Dead feature in this dataset |

**Why F2 and not accuracy?** In a dataset where ~98% of users are benign, a model that flags nobody scores 98% accuracy. F2-score penalizes missed threats (false negatives) twice as heavily as false alarms, reflecting the real cost asymmetry: a missed insider threat causes data loss; a false positive causes a 30-minute investigation.

---

## Dashboard

A Plotly-based SOC analyst interface (`dashboard.py`) visualizes:
- **User risk score timeline** — daily anomaly scores per user over the evaluation period
- **Anomaly heatmap** — all users × all days, color-coded by confirmed/review/gap status
- **Alert table** — sortable by threat category, ISO score, MAD flag ratio
- **SHAP waterfall viewer** — per-alert feature contribution breakdown

> 📸 Screenshots coming — dashboard requires the full pipeline run to generate output data.

---

## Repo Structure

```
├── src/
│   ├── detector.py          # Core dual-model detection engine
│   ├── extraction.py        # DuckDB feature extraction (logon/http/email)
│   ├── fusion.py            # Multi-source feature matrix fusion
│   ├── schema.py            # Master feature schema (single source of truth)
│   └── __init__.py
├── inference.py             # SHAP explainability layer + SOC report generation
├── dashboard.py             # Plotly analyst interface
├── diagnostics.py           # Model threshold diagnostic suite
├── exploration.ipynb        # EDA notebook
├── run_pipeline.ipynb       # End-to-end pipeline walkthrough
├── tests/
│   └── test_smoke.py
└── README.md
```

---

## Quickstart

### Requirements
```
Python 3.11+
duckdb, pandas, scikit-learn, shap, plotly, joblib, scipy
```

```bash
git clone https://github.com/sayantika-mlsec/insider-threat-detection-project.git
cd insider-threat-detection-project
pip install -r requirements.txt
```

### Data Setup
Download CERT Insider Threat Dataset r4.2 from [sei.cmu.edu/data-access](https://www.sei.cmu.edu/data-access) and place CSVs in `./data/`:
```
data/
├── logon.csv
├── http.csv
└── email.csv
```

### Run the Pipeline
Open `run_pipeline.ipynb` for the full end-to-end walkthrough, or run programmatically:

```python
from src.extraction import extract_logon_features, extract_http_features, extract_email_features
from src.fusion import fuse_feature_matrices
from src.detector import InsiderThreatDetector

# Extract features
logon_path = extract_logon_features("logon.csv", start_date="2010-01-01", end_date="2010-12-31")
http_path  = extract_http_features("http.csv",   start_date="2010-01-01", end_date="2011-12-31")
email_path = extract_email_features("email.csv", start_date="2010-01-01", end_date="2011-12-31")

# Fuse into master feature matrix
master_path = fuse_feature_matrices(logon_path, http_path, email_path,
                                     output_prefix="historical_baseline")

# Train
detector = InsiderThreatDetector(contamination=0.02)
model_path = detector.fit_baseline(master_path)

# Score live data
df_scores = detector.predict_live_traffic("features/live_master.parquet")
```

### Run Diagnostics
If alert counts look wrong, run the diagnostic suite before changing any model code:
```python
from diagnostics import run_full_diagnostic
run_full_diagnostic(detector, df_scores, df_live_features)
```

---

## Known Limitations & Production Gaps

These are documented honestly — not as failures, but as the gap between a research prototype and a production deployment. Each one has a known fix.

### 1. Threat Intelligence — Keyword Heuristic vs Live Feed
`keyword_match_indicator` currently matches against two hardcoded URL strings. In production, this column should be populated by joining HTTP logs against a daily-updated threat feed (URLhaus, OpenPhish, or an internal IOC database). A determined attacker who knows your detection system can trivially evade this feature by avoiding the two hardcoded strings.

### 2. Batch-Only Architecture — No Real-Time Path
The entire pipeline reads from parquet files and runs as a batch job. An insider exfiltrating data at 11pm will not be flagged until the next morning's batch run. Production insider threat systems require a streaming inference path (e.g., Kafka → feature extraction → real-time scoring) with sub-minute detection latency. This is the largest architectural gap between this prototype and a deployable system.

### 3. Role-Blind Global Baseline
When a new user has fewer than 5 days of history, they are scored against a single global population baseline computed across all users. This baseline mixes sysadmins, executives, interns, and contractors together. A new sysadmin doing legitimate high-volume work will be over-flagged; a new executive with unusually low activity will be under-flagged. Production systems stratify the global baseline by role or department.

### 4. Static Contamination Parameter
`contamination=0.02` was chosen based on the CERT dataset's known ~2% malicious user rate. In real enterprise deployments this rate is unknown, environment-specific, and changes over time. This parameter should be calibrated from ground-truth labels where available, or exposed as a deployment-time configuration with documented guidance on tuning.

### 5. Model Serialization Has No Dependency Manifest
The model is saved as a joblib pickle. The checksum verifies file integrity, but nothing records the Python version, scikit-learn version, or numpy version used at training time. Loading a model trained on sklearn 1.3 into a sklearn 1.5 environment may succeed silently while producing subtly different scores. Production ML artifacts should be versioned alongside a dependency manifest (e.g., a `requirements.txt` snapshot embedded in the artifact metadata).

### 6. No Protection Against SHAP Batch Overload
`explain_flagged_records()` processes all flagged records sequentially with no batch size limit. A data quality spike or policy change causing 10,000+ rows to be simultaneously flagged would block the alert pipeline for hours. Production deployments need a max batch size, async processing queue, or circuit breaker.

### 7. Disk Accumulation — SHAP Waterfall Plots
Each alert generates one `.png` waterfall plot with a deterministic filename based on (user, date, category). If a user's threat category changes between runs (e.g., `confirmed_threat` → `high_risk_review`), the old file is not overwritten — both accumulate. There is no TTL or cleanup job. On a production system processing thousands of daily alerts, `shap_plots/` will grow unboundedly without intervention.

---

## Future Work

- [ ] **Live threat feed integration** — replace `keyword_match_indicator` with URLhaus/OpenPhish API lookups
- [ ] **Role-stratified baselines** — use CERT role labels (`employee`, `IT admin`) to build separate global fallbacks per role class
- [ ] **Streaming inference** — Kafka consumer + River online learning for sub-minute detection latency
- [ ] **PyTorch Autoencoder comparison** — train a reconstruction-error autoencoder on normal traffic only; compare detection performance against Isolation Forest
- [ ] **File + Device log integration** — complete the remaining 2 of 5 CERT data sources
- [ ] **Model retraining pipeline** — scheduled retraining with Evidently AI drift detection to trigger retrain when behavioral distributions shift significantly

---

## Tech Stack

`Python 3.12` · `scikit-learn` · `DuckDB` · `SHAP` · `Plotly` · `joblib` · `scipy` · `pandas` · `numpy`

---

## Dataset Citation

> J. Glasser and B. Lindauer, "Bridging the Gap: A Pragmatic Approach to Generating Insider Threat Data," *2013 IEEE Security and Privacy Workshops*, San Francisco, CA, USA, 2013.

---

## Author

**SAYANTIKA**  
Aspiring ML Engineer | Actively building in Cybersecurity AI  
[GitHub](https://github.com/sayantika-mlsec/insider-threat-detection-project.git) . [Hugging Face](https://huggingface.co/spaces/sayantika-mlsec/insider-threat-soc-dashboard)

