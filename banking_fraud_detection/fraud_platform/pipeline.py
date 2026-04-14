"""
Core pipeline: data loading, preprocessing, training, evaluation, risk scoring, and prediction.

Designed for clarity in a university ML / fraud-detection assignment.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Configuration (easy to tune for demos)
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
N_SAMPLES = 5000
HIGH_RISK_THRESHOLD = 70.0
# All ``transaction_amount`` values in this project are Ugandan Shillings (UGX).
TRANSACTION_AMOUNT_CURRENCY = "UGX"
# Nominal USD-scale constants below are multiplied by this for synthetic UGX magnitudes (~demo FX).
UGX_PER_USD_DEMO = 3800.0
# Amount reference for sublinear (log) amount contribution to risk score (UGX)
AMOUNT_REFERENCE_FOR_RISK = 5000.0 * UGX_PER_USD_DEMO

# --- Hybrid ML layer (supervised LR + unsupervised anomaly) ---
# Fusion weights for hybrid risk_score (must sum to 1.0).
HYBRID_W_SUPERVISED = 0.52
HYBRID_W_ANOMALY = 0.30
HYBRID_W_AMOUNT = 0.18
# Isolation Forest: trained on legitimate-only rows; score_samples lower => more anomalous.
IF_N_ESTIMATORS = 200
IF_MAX_SAMPLES = 256
# If anomaly (normalized 0–1) is high, force review path even when p(fraud) is moderate.
ANOMALY_REVIEW_THRESHOLD = 0.72
# Optional: very high anomaly contributes an extra decline path with supervised still below DECLINE.
ANOMALY_DECLINE_THRESHOLD = 0.92

# --- Product-style routing (rules + model) ---
# Model: auto-decline only when fraud probability is very high (real banks rarely use p>0.5 alone).
DECLINE_PROBABILITY = 0.78
# Below this we may still approve; between this and DECLINE → step-up / review.
REVIEW_PROBABILITY = 0.40
# Risk score above this triggers step-up unless already declined.
REVIEW_RISK_SCORE = 52.0
# Cost-sensitive routing defaults (relative, synthetic units for demo tuning).
DEFAULT_COST_FALSE_APPROVE = 30.0
DEFAULT_COST_FALSE_DECLINE = 5.0
DEFAULT_COST_MANUAL_REVIEW = 1.0

# Policy examples (deterministic rules engine — common in production alongside ML).
POLICY_NEW_ACCOUNT_DAYS = 21
POLICY_NEW_ACCOUNT_REVIEW_AMOUNT = 750.0 * UGX_PER_USD_DEMO
POLICY_WIRE_REVIEW_AMOUNT = 8000.0 * UGX_PER_USD_DEMO
POLICY_CROSS_BORDER_REVIEW_AMOUNT = 3500.0 * UGX_PER_USD_DEMO
POLICY_LATE_HOUR_REVIEW_START = 21
POLICY_DECLINE_PRIOR_FRAUD_WIRE_AMOUNT = 12000.0 * UGX_PER_USD_DEMO
POLICY_DECLINE_SINGLE_TXN_CAP = 125000.0 * UGX_PER_USD_DEMO  # synthetic "hard stop" for loss control

# Simulated locations: US cities, remote channels, and Uganda (explicit cities for regional demos).
UGANDA_LOCATIONS = ("Kampala", "Entebbe", "Jinja", "Mbarara", "Gulu")
SIMULATED_LOCATIONS = (
    "NYC",
    "LA",
    "Chicago",
    "Miami",
    "Seattle",
    "Online",
    "Foreign",
    *UGANDA_LOCATIONS,
)
CROSS_BORDER_LOCATIONS = frozenset({"Foreign", "Online", *UGANDA_LOCATIONS})


def _build_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple account-behavior features using only prior activity.

    These are lightweight velocity-like features that mimic production fraud stacks:
    per-account recent transaction count and amount context.
    """
    if "account_id" not in df.columns or "event_time" not in df.columns:
        return df

    out = df.sort_values("event_time").copy()
    grp = out.groupby("account_id", sort=False)

    out["acct_txn_count_prev"] = grp.cumcount().astype(float)
    prev5_mean = (
        grp["transaction_amount"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        .fillna(0.0)
    )
    out["acct_amount_mean_prev5"] = prev5_mean
    out["amount_over_prev5_mean"] = out["transaction_amount"] / (out["acct_amount_mean_prev5"] + 1.0)
    out["hour_sin"] = np.sin((2.0 * np.pi * out["transaction_time"]) / 24.0)
    out["hour_cos"] = np.cos((2.0 * np.pi * out["transaction_time"]) / 24.0)
    return out


def load_data(
    n_samples: int = N_SAMPLES,
    random_state: int = RANDOM_STATE,
    save_path: str | None = None,
) -> pd.DataFrame:
    """
    Create a simulated banking transactions dataset.

    ``transaction_amount`` is in **UGX** (Ugandan Shillings). In production you would read
    from CSV/DB; here we synthesize realistic-looking rows and inject fraud using simple
    rules + noise so models have patterns to learn.
    """
    rng = np.random.default_rng(random_state)

    # Categorical pools
    locations = list(SIMULATED_LOCATIONS)
    tx_types = ["Wire", "Card", "ACH", "ATM", "Online_Payment"]

    n = n_samples
    transaction_amount = (
        rng.lognormal(mean=4.0, sigma=1.2, size=n).astype(float) * UGX_PER_USD_DEMO
    )
    transaction_time = rng.integers(0, 24, size=n)  # hour of day
    location = rng.choice(locations, size=n)
    account_age_days = rng.integers(1, 3650, size=n)
    transaction_type = rng.choice(tx_types, size=n)
    previous_fraud_history = rng.integers(0, 2, size=n)
    n_accounts = max(200, n // 8)
    account_ids = np.array([f"ACCT_{i:05d}" for i in range(n_accounts)])
    account_id = rng.choice(account_ids, size=n, replace=True)
    interarrival_minutes = rng.integers(1, 12, size=n)
    minute_offsets = np.cumsum(interarrival_minutes)
    event_time = pd.Timestamp("2025-01-01 00:00:00") + pd.to_timedelta(
        minute_offsets, unit="m"
    )

    df = pd.DataFrame(
        {
            "event_time": event_time,
            "account_id": account_id,
            "transaction_amount": transaction_amount,
            "transaction_time": transaction_time,
            "location": location,
            "account_age": account_age_days,
            "transaction_type": transaction_type,
            "previous_fraud_history": previous_fraud_history,
        }
    )

    # --- Fraud label simulation (interpretable rules + random fraud) ---
    fraud_score = np.zeros(n, dtype=float)

    # Higher amounts are riskier
    fraud_score += (transaction_amount > np.percentile(transaction_amount, 92)) * 2.0
    # New accounts are riskier
    fraud_score += (account_age_days < 60) * 1.5
    # Prior fraud flag
    fraud_score += previous_fraud_history * 2.0
    # Foreign / online / Uganda (cross-border style risk in this synthetic US-centric bank)
    fraud_score += np.isin(location, list(CROSS_BORDER_LOCATIONS)) * 1.2
    # Late night activity
    fraud_score += np.isin(transaction_time, [0, 1, 2, 3, 22, 23]) * 0.8
    # Wire transfers with high amount
    wire_high_ugx = 3000.0 * UGX_PER_USD_DEMO
    fraud_score += ((transaction_type == "Wire") & (transaction_amount > wire_high_ugx)) * 1.5

    # Random baseline fraud
    fraud_prob = 1 / (1 + np.exp(-(fraud_score - 3.0)))
    is_fraud = (rng.random(n) < fraud_prob).astype(int)

    df["is_fraud"] = is_fraud

    # Sprinkle missing values (small fraction) to demonstrate imputation
    miss_idx = rng.choice(n, size=int(0.02 * n), replace=False)
    df.loc[miss_idx[: len(miss_idx) // 2], "transaction_amount"] = np.nan
    df.loc[miss_idx[len(miss_idx) // 2 :], "account_age"] = np.nan
    df = _build_behavioral_features(df)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        df.to_csv(save_path, index=False)

    return df


def preprocess_data(
    df: pd.DataFrame,
    target_col: str = "is_fraud",
) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer, list[str], list[str]]:
    """
    Handle missing values, define feature groups, and build a preprocessor
    (scaling for numeric, one-hot encoding for categoricals).

    Returns X, y, preprocessor (unfitted), numeric feature names, categorical names.
    """
    df = df.copy()

    # Separate target
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    # Column groups (imputation happens inside the sklearn Pipeline so new rows are safe)
    num_cols = [
        "transaction_amount",
        "transaction_time",
        "account_age",
        "previous_fraud_history",
        "acct_txn_count_prev",
        "acct_amount_mean_prev5",
        "amount_over_prev5_mean",
        "hour_sin",
        "hour_cos",
    ]
    cat_cols = ["location", "transaction_type"]

    # Optional: fill for exploratory CSV export consistency (Pipeline still imputes at fit time)
    for col in num_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    for col in cat_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna("Unknown")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    return X, y, preprocessor, num_cols, cat_cols


def train_anomaly_detector(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
) -> Tuple[Pipeline, float, float]:
    """
    Unsupervised branch of the hybrid layer: Isolation Forest fit on **legitimate**
    training rows only (one-class style). Lower ``score_samples`` ⇒ more anomalous.

    Calibration percentiles (q05, q95) come from the same legitimate slice so we can
    map raw scores to a stable 0–1 ``anomaly_normalized`` scale for fusion with LR.
    """
    legit_mask = y_train.astype(int).values == 0
    X_legit = X_train.loc[legit_mask]
    if len(X_legit) < 50:
        X_legit = X_train

    max_samples = min(IF_MAX_SAMPLES, max(1, len(X_legit)))
    ano_pipe = Pipeline(
        steps=[
            ("prep", clone(preprocessor)),
            (
                "iforest",
                IsolationForest(
                    n_estimators=IF_N_ESTIMATORS,
                    max_samples=max_samples,
                    random_state=RANDOM_STATE,
                    contamination="auto",
                ),
            ),
        ]
    )
    ano_pipe.fit(X_legit)
    raw = ano_pipe.score_samples(X_legit)
    q05, q95 = float(np.percentile(raw, 5)), float(np.percentile(raw, 95))
    if q95 - q05 < 1e-6:
        q05 -= 1e-3
        q95 += 1e-3
    return ano_pipe, q05, q95


def anomaly_score_normalized(
    anomaly_pipeline: Pipeline,
    calib: Tuple[float, float],
    X: pd.DataFrame,
) -> np.ndarray:
    """Vector of 0–1 anomaly scores (higher = more anomalous) for each row of ``X``."""
    q05, q95 = calib
    raw = anomaly_pipeline.score_samples(X)
    span = q95 - q05 + 1e-9
    t = (q95 - raw) / span
    return np.clip(t, 0.0, 1.0)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
) -> Dict[str, Any]:
    """
    Train Logistic Regression and a Decision Tree on the training split.

    Logistic Regression gives well-behaved probabilities for risk scoring.
    The tree is useful for non-linear patterns and feature-importance plots.
    """
    models: Dict[str, Any] = {}

    lr_pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    lr_pipe.fit(X_train, y_train)
    models["logistic_regression"] = lr_pipe

    # Tree uses same preprocessor via a fresh clone pipeline
    tree_pipe = Pipeline(
        steps=[
            ("prep", clone(preprocessor)),
            (
                "clf",
                DecisionTreeClassifier(
                    max_depth=8,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    tree_pipe.fit(X_train, y_train)
    models["decision_tree"] = tree_pipe

    ano_pipe, q05, q95 = train_anomaly_detector(X_train, y_train, preprocessor)
    models["anomaly_detector"] = ano_pipe
    models["anomaly_calib"] = (q05, q95)
    models["routing_thresholds"] = tune_routing_thresholds(lr_pipe, X_train, y_train)

    return models


def tune_routing_thresholds(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    cost_false_approve: float = DEFAULT_COST_FALSE_APPROVE,
    cost_false_decline: float = DEFAULT_COST_FALSE_DECLINE,
    cost_manual_review: float = DEFAULT_COST_MANUAL_REVIEW,
) -> Dict[str, float]:
    """
    Tune review/decline probability thresholds on a temporal holdout from training data.

    Objective: reduce expected fraud leakage (false approve) while controlling decline/review
    burden, using a simple weighted cost function.
    """
    if len(X_train) < 250 or not hasattr(model, "predict_proba"):
        return {
            "review_probability": REVIEW_PROBABILITY,
            "decline_probability": DECLINE_PROBABILITY,
        }

    n_val = max(100, int(0.2 * len(X_train)))
    X_val = X_train.iloc[-n_val:]
    y_val = y_train.iloc[-n_val:].astype(int).to_numpy()
    probs = model.predict_proba(X_val)[:, 1]

    best_loss = float("inf")
    best = (REVIEW_PROBABILITY, DECLINE_PROBABILITY)

    for decline in np.linspace(0.60, 0.95, 36):
        for review in np.linspace(0.15, decline - 0.05, 40):
            approved = probs < review
            declined = probs >= decline
            reviewed = (~approved) & (~declined)

            false_approves = int(np.sum((y_val == 1) & approved))
            false_declines = int(np.sum((y_val == 0) & declined))
            review_count = int(np.sum(reviewed))
            loss = (
                cost_false_approve * false_approves
                + cost_false_decline * false_declines
                + cost_manual_review * review_count
            )
            if loss < best_loss:
                best_loss = loss
                best = (float(review), float(decline))

    return {
        "review_probability": best[0],
        "decline_probability": best[1],
    }


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "model",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compute accuracy, precision, recall, confusion matrix, and print a short explanation.

    `model` is expected to be a sklearn Pipeline with .predict and (for LR) .predict_proba.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    if verbose:
        print(f"\n--- Evaluation: {model_name} ---")
        print(f"Accuracy:  {acc:.4f}  (fraction of all predictions that are correct)")
        print(f"Precision: {prec:.4f}  (of predicted fraud cases, how many are truly fraud)")
        print(f"Recall:    {rec:.4f}  (of all real fraud cases, how many we catch)")
        print("Confusion matrix [[TN, FP],[FN, TP]]:")
        print(cm)
        print("\nClassification report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        explanation = (
            "Precision matters when false alarms are costly; recall matters when missing "
            "fraud is costly. For fraud, teams often prefer higher recall at some precision cost."
        )
        print(f"\nNote: {explanation}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": cm,
        "y_pred": y_pred,
    }


def risk_score(
    fraud_probability: float,
    transaction_amount: float,
    anomaly_normalized: float | None = None,
    *,
    w_prob: float = 0.72,
    w_amount: float = 0.28,
) -> float:
    """
    Map fraud probability and transaction size to a 0–100 risk score.

    If ``anomaly_normalized`` is provided (0–1 from the Isolation Forest branch), use the
    **hybrid** fusion weights (supervised + anomaly + amount). Otherwise keep the legacy
    two-term blend (probability + amount only).

    Uses a sublinear (log) amount term so extreme amounts alone do not peg the score
    at 100 unless the model probability is also high — closer to real risk engines.
    """
    p = float(np.clip(fraud_probability, 0.0, 1.0))
    ref = AMOUNT_REFERENCE_FOR_RISK
    amount_factor = float(np.clip(np.log1p(max(transaction_amount, 0.0)) / np.log1p(ref), 0.0, 1.0))

    if anomaly_normalized is None:
        score = 100.0 * (w_prob * p + w_amount * amount_factor)
    else:
        a = float(np.clip(anomaly_normalized, 0.0, 1.0))
        score = 100.0 * (
            HYBRID_W_SUPERVISED * p
            + HYBRID_W_ANOMALY * a
            + HYBRID_W_AMOUNT * amount_factor
        )
    return float(np.clip(score, 0.0, 100.0))


def evaluate_policies(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic rules (policy) that run beside the ML model.

    Returns reason codes and a coarse severity: None | \"review\" | \"decline\".
    Decline reasons override review-only flags (loss / compliance style simulation).
    """
    amt = float(transaction["transaction_amount"])
    loc = str(transaction["location"])
    acct_age = int(transaction["account_age"])
    prev = int(transaction["previous_fraud_history"])
    tx_type = str(transaction["transaction_type"])
    hour = int(transaction["transaction_time"])

    reasons: list[str] = []
    worst: str | None = None

    def add(code: str, level: str) -> None:
        reasons.append(code)
        nonlocal worst
        if level == "decline":
            worst = "decline"
        elif level == "review" and worst != "decline":
            worst = "review"

    if amt >= POLICY_DECLINE_SINGLE_TXN_CAP:
        add("POLICY_SINGLE_TXN_LIMIT", "decline")
    if prev == 1 and tx_type == "Wire" and amt > POLICY_DECLINE_PRIOR_FRAUD_WIRE_AMOUNT:
        add("POLICY_PRIOR_FRAUD_LARGE_WIRE", "decline")

    if worst != "decline":
        if acct_age < POLICY_NEW_ACCOUNT_DAYS and amt > POLICY_NEW_ACCOUNT_REVIEW_AMOUNT:
            add("POLICY_NEW_ACCOUNT_ELEVATED_AMOUNT", "review")
        if tx_type == "Wire" and amt > POLICY_WIRE_REVIEW_AMOUNT:
            add("POLICY_WIRE_ABOVE_REVIEW_THRESHOLD", "review")
        if loc in CROSS_BORDER_LOCATIONS and amt > POLICY_CROSS_BORDER_REVIEW_AMOUNT:
            add("POLICY_CROSS_CHANNEL_ELEVATED_AMOUNT", "review")
        if hour >= POLICY_LATE_HOUR_REVIEW_START:
            add("POLICY_LATE_HOUR_TRANSACTION", "review")

    return {"reason_codes": reasons, "policy_severity": worst}


def _route_decision(
    fraud_probability: float,
    fraud_prediction: int,
    risk: float,
    policy: Dict[str, Any],
    anomaly_normalized: float | None = None,
    routing_thresholds: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """
    Combine model scores + policy into a final disposition (like auth + fraud stacks).

    Order: hard policy decline → high-confidence model decline → step-up/review → approve.
    """
    sev = policy["policy_severity"]
    codes = list(policy["reason_codes"])
    review_probability = REVIEW_PROBABILITY
    decline_probability = DECLINE_PROBABILITY
    if routing_thresholds:
        review_probability = float(
            routing_thresholds.get("review_probability", REVIEW_PROBABILITY)
        )
        decline_probability = float(
            routing_thresholds.get("decline_probability", DECLINE_PROBABILITY)
        )

    if sev == "decline":
        return {
            "decision": "declined",
            "decision_source": "policy",
            "reason_codes": codes,
        }
    if fraud_probability >= decline_probability:
        out_codes = codes + ["MODEL_HIGH_FRAUD_PROBABILITY"]
        return {
            "decision": "declined",
            "decision_source": "model",
            "reason_codes": out_codes,
        }

    if (
        anomaly_normalized is not None
        and anomaly_normalized >= ANOMALY_DECLINE_THRESHOLD
        and fraud_probability >= review_probability
    ):
        out_codes = codes + ["MODEL_HYBRID_EXTREME_ANOMALY_WITH_ELEVATED_PROB"]
        return {
            "decision": "declined",
            "decision_source": "hybrid",
            "reason_codes": out_codes,
        }

    model_review = (
        fraud_prediction == 1
        or fraud_probability >= review_probability
        or risk >= REVIEW_RISK_SCORE
        or (
            anomaly_normalized is not None
            and anomaly_normalized >= ANOMALY_REVIEW_THRESHOLD
        )
    )
    policy_review = sev == "review"

    if policy_review or model_review:
        extra: list[str] = []
        if fraud_prediction == 1 and fraud_probability < decline_probability:
            extra.append("MODEL_FRAUD_FLAG_LOW_CONFIDENCE_REVIEW")
        if fraud_probability >= review_probability:
            extra.append("MODEL_ELEVATED_FRAUD_PROBABILITY")
        if risk >= REVIEW_RISK_SCORE:
            extra.append("MODEL_ELEVATED_RISK_SCORE")
        if (
            anomaly_normalized is not None
            and anomaly_normalized >= ANOMALY_REVIEW_THRESHOLD
        ):
            extra.append("MODEL_ELEVATED_ANOMALY_SCORE")
        return {
            "decision": "step_up",
            "decision_source": "policy" if policy_review and not model_review else "combined",
            "reason_codes": codes + extra,
        }

    return {
        "decision": "approved",
        "decision_source": "model",
        "reason_codes": codes,
    }


def _format_alerts(
    decision: str,
    risk: float,
    fraud_prediction: int,
    anomaly_normalized: float | None = None,
    reason_codes: list[str] | None = None,
) -> list[str]:
    """Human-readable lines for console / API responses."""
    lines: list[str] = []
    if anomaly_normalized is not None and anomaly_normalized >= ANOMALY_REVIEW_THRESHOLD:
        lines.append(
            f"Anomaly layer: unusual-pattern score {anomaly_normalized:.2f} (0-1, higher = rarer)"
        )
    if reason_codes and "POLICY_LATE_HOUR_TRANSACTION" in reason_codes:
        lines.append("⏰ Late-hour transaction (21:00-23:59) — policy flag added for manual review context")
    if risk > HIGH_RISK_THRESHOLD:
        lines.append("⚠️ High risk score — flagged for monitoring / review context")
    if decision == "declined":
        lines.append("🚫 Transaction declined (policy or high-confidence fraud score)")
    elif decision == "step_up":
        lines.append("📋 Step-up required: send to manual review / OTP / call-back queue")
    else:
        lines.append("✅ Transaction approved for processing")
    if fraud_prediction == 1 and decision != "declined":
        lines.append("ℹ️ Model fraud class with probability below auto-decline — routed to review")
    return lines


def predict_transaction(
    model: Any,
    transaction: Dict[str, Any],
    feature_columns: list[str],
    *,
    anomaly_detector: Pipeline | None = None,
    anomaly_calib: Tuple[float, float] | None = None,
    routing_thresholds: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """
    Score one transaction: ML + policy routing + risk + user-facing alerts.

    ``decision`` is product-like: approved | step_up | declined.
    ``reason_codes`` may list both policy and model tags for audit (even if the final
    ``decision_source`` is only \"model\" or \"policy\").
    """
    row = pd.DataFrame([{k: transaction.get(k, np.nan) for k in feature_columns}])
    for feature in ("acct_txn_count_prev", "acct_amount_mean_prev5", "amount_over_prev5_mean"):
        if feature in row.columns and pd.isna(row.at[0, feature]):
            row.at[0, feature] = 0.0
    if "hour_sin" in row.columns and pd.isna(row.at[0, "hour_sin"]):
        hour = float(transaction.get("transaction_time", 0.0))
        row.at[0, "hour_sin"] = np.sin((2.0 * np.pi * hour) / 24.0)
    if "hour_cos" in row.columns and pd.isna(row.at[0, "hour_cos"]):
        hour = float(transaction.get("transaction_time", 0.0))
        row.at[0, "hour_cos"] = np.cos((2.0 * np.pi * hour) / 24.0)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row)[0]
        clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
        classes = getattr(clf, "classes_", np.array([0, 1]))
        fraud_idx = int(np.where(classes == 1)[0][0])
        fraud_probability = float(proba[fraud_idx])
        y_hat = int(model.predict(row)[0])
    else:
        fraud_probability = float(model.predict(row)[0])
        y_hat = int(fraud_probability)

    amount = float(transaction["transaction_amount"])

    anomaly_raw: float | None = None
    anomaly_norm: float | None = None
    hybrid_active = anomaly_detector is not None and anomaly_calib is not None
    if hybrid_active:
        anomaly_raw = float(anomaly_detector.score_samples(row)[0])
        span = anomaly_calib[1] - anomaly_calib[0] + 1e-9
        anomaly_norm = float(
            np.clip((anomaly_calib[1] - anomaly_raw) / span, 0.0, 1.0)
        )
        rscore = risk_score(fraud_probability, amount, anomaly_norm)
    else:
        rscore = risk_score(fraud_probability, amount)

    policy = evaluate_policies(transaction)
    routed = _route_decision(
        fraud_probability,
        y_hat,
        rscore,
        policy,
        anomaly_normalized=anomaly_norm,
        routing_thresholds=routing_thresholds,
    )

    alerts = _format_alerts(
        routed["decision"],
        rscore,
        y_hat,
        anomaly_norm,
        routed["reason_codes"],
    )

    out: Dict[str, Any] = {
        "fraud_prediction": y_hat,
        "fraud_probability": fraud_probability,
        "risk_score": rscore,
        "decision": routed["decision"],
        "decision_source": routed["decision_source"],
        "reason_codes": routed["reason_codes"],
        "policy_severity": policy["policy_severity"],
        "alerts": alerts,
        "hybrid_layer": hybrid_active,
    }
    if hybrid_active:
        out["anomaly_score_raw"] = anomaly_raw
        out["anomaly_score_normalized"] = anomaly_norm
    return out


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prefer temporal split by ``event_time`` (fallback to stratified random split)."""
    if "event_time" in X.columns:
        order = X["event_time"].sort_values().index
        X_sorted = X.loc[order]
        y_sorted = y.loc[order]
        n_test = max(1, int(round(test_size * len(X_sorted))))
        split_idx = len(X_sorted) - n_test
        return (
            X_sorted.iloc[:split_idx].copy(),
            X_sorted.iloc[split_idx:].copy(),
            y_sorted.iloc[:split_idx].copy(),
            y_sorted.iloc[split_idx:].copy(),
        )

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def tree_feature_importance_named(model_pipeline: Pipeline) -> pd.Series:
    """Readable feature names from the ColumnTransformer + tree importances."""
    prep: ColumnTransformer = model_pipeline.named_steps["prep"]
    tree: DecisionTreeClassifier = model_pipeline.named_steps["clf"]
    feature_names = list(prep.get_feature_names_out())
    imp = pd.Series(tree.feature_importances_, index=feature_names)
    return imp.sort_values(ascending=False)
