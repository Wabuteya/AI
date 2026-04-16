# PARALLAX Bank Fraud Detection System Report

## 1) System Purpose

This project is an end-to-end fraud analytics system for banking transactions. It combines:

- Synthetic transaction data generation
- Feature engineering and preprocessing
- Supervised and unsupervised machine learning
- Policy/rules-based controls
- Decision routing (`approved`, `step_up`, `declined`)
- Two user interfaces:
  - Command-line workflow (`main.py`)
  - Streamlit dashboard (`app.py`)

The goal is to simulate how a practical fraud platform works by blending model predictions with deterministic business policy.


## 2) High-Level Architecture

Core code is organized into:

- `fraud_platform/pipeline.py` — central business and ML logic
- `main.py` — CLI entry point and scripted demos
- `app.py` — interactive Streamlit fraud console
- `fraud_platform/__init__.py` — package exports
- `.streamlit/config.toml` — Streamlit UI defaults
- `requirements.txt` — runtime dependencies

The architecture is deliberately modular:

1. **Data layer**: `load_data()` creates and optionally saves simulated transactions.
2. **Feature layer**: `_build_behavioral_features()` and `preprocess_data()` engineer and prepare features.
3. **Model layer**: `train_model()` trains:
   - Logistic Regression (primary supervised probability model)
   - Decision Tree (interpretable benchmark + feature importances)
   - Isolation Forest (anomaly branch)
4. **Scoring/risk layer**: `predict_transaction()` computes fraud probability, anomaly signal, and fused risk score.
5. **Policy/routing layer**: `evaluate_policies()` + `_route_decision()` produce the final decision and reason codes.
6. **Presentation layer**: CLI/Streamlit expose metrics, plots, and single-transaction scoring.


## 3) Data Model and Synthetic Data Generation

### 3.1 Generated fields

`load_data()` creates transaction records with:

- `event_time` (monotonic synthetic timestamp)
- `account_id`
- `transaction_amount` (UGX)
- `transaction_time` (hour 0-23)
- `location`
- `account_age` (days)
- `transaction_type`
- `previous_fraud_history`
- `is_fraud` (label)

### 3.2 Label simulation logic

Fraud labels are generated from a latent `fraud_score` based on interpretable risk factors:

- Very high transaction amount
- Newer accounts
- Prior fraud history
- Cross-border/remote style locations
- Late-night activity
- High-value wire behavior

The score is transformed to probability (sigmoid), then sampled into binary fraud labels.

### 3.3 Data quality simulation

Small amounts of missingness are injected into numeric columns to exercise imputation paths.


## 4) Feature Engineering and Preprocessing

### 4.1 Behavioral features

`_build_behavioral_features()` adds lightweight account behavior signals:

- `acct_txn_count_prev`
- `acct_amount_mean_prev5`
- `amount_over_prev5_mean`
- `hour_sin`, `hour_cos` (cyclical hour representation)

These mimic production-style velocity/context signals while staying simple.

### 4.2 Preprocessing pipeline

`preprocess_data()` builds a `ColumnTransformer`:

- Numeric pipeline:
  - median imputation
  - standardization
- Categorical pipeline:
  - one-hot encoding (`handle_unknown="ignore"`)

The function returns:

- `X`, `y`
- unfitted `preprocessor`
- `num_cols`, `cat_cols`


## 5) Training Pipeline

`train_model()` trains three branches:

1. **Logistic Regression**
   - weighted classes (`class_weight="balanced"`)
   - high `max_iter` for convergence
   - used as primary fraud probability source

2. **Decision Tree**
   - moderate depth (`max_depth=8`)
   - class-balanced
   - used for comparison and feature-importance reporting

3. **Isolation Forest (anomaly)**
   - fit on legitimate-only training rows when possible
   - generates anomaly raw scores
   - calibrated to normalized [0,1] anomaly scale using 5th/95th percentiles

It also tunes decision thresholds through `tune_routing_thresholds()`.


## 6) Threshold Tuning and Cost-Based Routing Calibration

`tune_routing_thresholds()` performs a grid search on training temporal holdout probabilities:

- Chooses `review_probability` and `decline_probability`
- Minimizes weighted loss from:
  - false approves (highest cost)
  - false declines
  - manual reviews

This simulates operational trade-offs between fraud leakage, customer friction, and review queue load.


## 7) Evaluation and Explainability Outputs

`evaluate_model()` computes:

- Accuracy
- Precision
- Recall
- Confusion matrix
- Classification report

For CLI use, it prints short metric interpretation guidance, emphasizing fraud-specific precision/recall trade-offs.

`tree_feature_importance_named()` maps Decision Tree feature importances to readable transformed feature names.


## 8) Hybrid Risk Scoring Logic

`risk_score()` outputs a 0-100 risk score:

- Baseline mode: supervised probability + sublinear amount term
- Hybrid mode: supervised + anomaly + amount

Amount contribution is logarithmic, so very large amounts alone do not saturate risk unless model signals agree.

Key design intent:

- Capture fraud likelihood (`fraud_probability`)
- Add unusual-pattern pressure (`anomaly_normalized`)
- Retain transaction-size context (`transaction_amount`)


## 9) Policy Engine and Decision Routing

### 9.1 Policy evaluation

`evaluate_policies()` assigns reason codes and severity:

- Hard decline rules (e.g., single-transaction cap, prior-fraud large wire)
- Review rules (e.g., new-account elevated amount, large wire, cross-channel elevated amount)

### 9.2 Final decision routing

`_route_decision()` combines policy + model + risk:

Order of precedence:

1. Policy hard decline
2. Model high-probability decline
3. Hybrid extreme-anomaly decline (conditional)
4. Step-up/review if model/policy review criteria met
5. Approve otherwise

Every non-trivial decision includes `reason_codes` and `decision_source` for auditability.


## 10) Single-Transaction Inference Contract

`predict_transaction()` is the operational scoring entry point.

Input:

- trained model
- transaction dictionary
- `feature_columns`
- optional anomaly model/calibration and routing thresholds

Output payload includes:

- `fraud_prediction` (0/1)
- `fraud_probability`
- `risk_score`
- `decision` (`approved`, `step_up`, `declined`)
- `decision_source`
- `reason_codes`
- `policy_severity`
- `alerts`
- hybrid metadata (`anomaly_score_raw`, `anomaly_score_normalized`) when active


## 11) CLI Application (`main.py`)

`main.py` orchestrates full workflow:

1. Generate/load synthetic data and save CSV
2. Preprocess and split (temporal preferred)
3. Train all models
4. Evaluate LR + Tree
5. Save plots:
   - confusion matrix
   - feature importance
6. Print sample hybrid scores on test rows
7. Optional deterministic demos via `--demo-tx`
8. Interactive transaction entry (default unless `--no-interactive`)

The CLI is suitable for quick reproducible runs and assignment-style reporting.


## 12) Streamlit Application (`app.py`)

The web console offers a user-facing fraud operations dashboard:

- Sidebar controls for data volume, seed, output path, retraining
- KPI cards (fraud rate, recall, thresholds)
- Tabs:
  - **Overview**: metrics, confusion matrix, feature importances, sample scores
  - **Transaction Scoring**: form-based transaction entry, decision banner, gauge, alerts
  - **Data Explorer**: dataset preview and CSV download
- Download options for plots, scores, and decision outputs

Training artifacts and models are cached in `st.session_state`.


## 13) Configuration and Dependencies

### 13.1 Runtime dependencies

From `requirements.txt`:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `streamlit`

### 13.2 Streamlit defaults

`.streamlit/config.toml` defines theme colors and disables usage stats collection.


## 14) End-to-End Data Flow

1. User runs CLI or Streamlit
2. Data synthesized (`load_data`)
3. Behavioral features added
4. Preprocessing configured
5. Train/test split performed (temporal)
6. Models trained (LR, Tree, Isolation Forest)
7. Routing thresholds tuned
8. Evaluation metrics/plots generated
9. Transaction scored through:
   - supervised probability
   - anomaly normalization
   - fused risk score
   - policy checks
   - routing decision
10. Decision package returned to user/UI with reason codes and alerts


## 15) Operational Strengths and Practical Notes

Strengths:

- Realistic hybrid approach (supervised + anomaly + policy)
- Explainable outputs (reason codes, source attribution, metrics)
- Temporal split support
- Downloadable artifacts for reporting
- Two interfaces for both technical and non-technical users

Practical constraints:

- Data is synthetic, not production banking telemetry
- Rules and costs are illustrative and static
- No model persistence/version registry in current code
- No external API service wrapper in this project layout


## 16) How to Run

CLI:

- `python main.py`
- `python main.py --demo-tx --no-interactive`

Streamlit:

- `streamlit run app.py`


## 17) Core AI Technologies Implemented

This project now implements the key technology blocks described in enterprise fraud platforms, mapped directly to code in this system.

### 17.1 Real-time transaction monitoring and instant scoring

- `monitor_transaction_realtime()` wraps scoring and returns `processing_latency_ms`, enabling millisecond-scale monitoring metadata for each decision.
- `predict_transaction()` computes fraud probability, anomaly score, contextual risk uplift, policy checks, and final routing in a single transaction flow.
- Final output includes an auditable package (`decision`, `decision_source`, `reason_codes`, `alerts`) for operations use.

### 17.2 Machine learning (supervised + anomaly hybrid)

- Supervised branch: Logistic Regression (`train_model`) estimates `fraud_probability`.
- Unsupervised branch: Isolation Forest (`train_anomaly_detector`) detects unusual behavior from legitimate baseline patterns.
- Fusion: `risk_score()` combines supervised probability, anomaly signal, and amount context into a 0-100 hybrid risk score.

### 17.3 Graph analytics (entity-link risk)

- `graph_link_risk()` provides graph-style relationship analysis over connected entities such as:
  - `account_id`
  - `device_id`
  - `beneficiary_id`
- The function checks links to recently observed fraudulent neighborhoods and returns both:
  - `graph_risk_score`
  - graph-specific reason codes for explainability.

### 17.4 NLP on unstructured payment instructions

- `nlp_instruction_risk()` processes unstructured payment text (memo/instruction fields) and flags suspicious BEC-like phrasing.
- Matched phrases are converted into:
  - `nlp_risk_score`
  - explicit reason codes (`NLP_SUSPICIOUS_PHRASE:...`) for analyst review.

### 17.5 Behavioral biometrics

- `behavioral_biometrics_risk()` computes a behavior anomaly signal from user-interaction deviations:
  - typing cadence deviation (`typing_cadence_z`)
  - mouse velocity deviation (`mouse_velocity_z`)
  - session pattern deviation (`session_deviation_z`)
- Output is a normalized `behavior_risk_score` plus behavior-specific reason tags.

### 17.6 Human-in-the-loop learning workflow

- `apply_analyst_feedback()` enables threshold adaptation after analyst outcomes:
  - false positive feedback -> slightly raises review/decline thresholds
  - missed fraud feedback -> slightly lowers thresholds
- In Streamlit, this is exposed through analyst controls ("Mark as False Positive" / "Mark as Missed Fraud"), supporting continuous operational tuning.

### 17.7 Decision orchestration across all signals

- `_route_decision()` handles policy + model routing.
- `_context_escalation()` adds contextual escalation so strong graph/NLP/behavior signals can move an otherwise approved transaction to `step_up`.
- This supports a practical layered strategy: model confidence, deterministic policy, and context intelligence all contribute to final action.

### 17.8 JPMC-inspired enterprise operations layer (educational simulation)

To align with enterprise-bank operating patterns, the real-time monitor now produces additional operational metadata:

- `enterprise_action`: `allow_transaction`, `challenge_and_review`, or `block_transaction`
- `review_priority`: queue level (`P1`..`P4`) based on risk severity
- `analyst_review_required`: explicit human-in-the-loop flag
- `case_id`: generated analyst case reference for triage workflow
- `model_lineage`: model version tag for governance-style traceability

Important note: this is an **educational, JPMC-inspired architecture**, not JPMorganChase proprietary code or internal platform replication.

## 18) Conclusion

This system demonstrates a complete fraud decision stack: simulated data generation, ML model training, anomaly detection, policy enforcement, graph/NLP/behavior context scoring, human-in-the-loop threshold refinement, and explainable transaction routing. It is intentionally educational but architected in a way that mirrors real fraud platforms, where deterministic controls and probabilistic models work together to manage fraud loss and customer friction.
