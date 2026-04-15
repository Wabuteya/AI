#!/usr/bin/env python3
"""
AI-Based Banking Transaction Simulator with Real-Time Fraud Detection

Pipeline:
Transaction -> Validation -> Fraud Detection -> Risk Scoring -> Decision -> Alert -> Logging
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42


@dataclass
class UserProfile:
    account_id: str
    home_location: str
    average_transaction_amount: float
    usual_transaction_hours: tuple[int, int]
    primary_device: str


def get_user_profile(account_id: str) -> UserProfile:
    """
    Return a simulated account behavior profile.
    """
    rng = np.random.default_rng(abs(hash(account_id)) % (2**32))
    home_locations = ["Kampala", "Nairobi", "London", "Dubai", "Johannesburg"]
    location = str(rng.choice(home_locations))
    avg_amount = float(rng.integers(40_000, 400_000))
    start_hour = int(rng.integers(6, 11))
    end_hour = int(rng.integers(18, 23))
    primary_device = f"DEV_{int(rng.integers(1_000, 9_999))}"
    return UserProfile(
        account_id=account_id,
        home_location=location,
        average_transaction_amount=avg_amount,
        usual_transaction_hours=(start_hour, end_hour),
        primary_device=primary_device,
    )


def generate_transaction(
    scenario: str,
    profile: UserProfile,
    rng: np.random.Generator,
    recent_transactions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Generate one realistic transaction in a selected scenario.
    """
    tx_types = ["ATM", "POS", "Online"]
    now = datetime.now()

    if scenario == "normal":
        amount = float(max(1_000, rng.normal(profile.average_transaction_amount, 20_000)))
        location = profile.home_location if rng.random() < 0.9 else "Nairobi"
        hour = int(rng.integers(profile.usual_transaction_hours[0], profile.usual_transaction_hours[1] + 1))
        device = profile.primary_device if rng.random() < 0.9 else f"DEV_{int(rng.integers(10_000, 99_999))}"
    elif scenario == "suspicious":
        amount = float(max(1_000, rng.normal(profile.average_transaction_amount * 2.3, 70_000)))
        location = "London" if profile.home_location != "London" else "Dubai"
        hour = int(rng.choice([1, 2, 3, 23]))
        device = f"DEV_{int(rng.integers(10_000, 99_999))}"
    else:  # fraud_attack
        amount = float(max(1_000, rng.normal(profile.average_transaction_amount * 4.0, 160_000)))
        location = str(rng.choice(["London", "Dubai", "Lagos", "HongKong"]))
        hour = int(rng.choice([0, 1, 2, 3, 4, 23]))
        device = f"DEV_{int(rng.integers(10_000, 99_999))}"

    # Simulate rapid-fire pattern by forcing same/near timestamp in suspicious/fraud modes.
    if scenario in ("suspicious", "fraud_attack") and recent_transactions and rng.random() < 0.45:
        ts = recent_transactions[-1]["transaction_time"] + timedelta(seconds=int(rng.integers(3, 35)))
    else:
        ts = now.replace(hour=hour, minute=int(rng.integers(0, 60)), second=int(rng.integers(0, 60)))

    return {
        "account_id": profile.account_id,
        "transaction_amount": round(amount, 2),
        "transaction_time": ts,
        "location": location,
        "device_id": device,
        "transaction_type": str(rng.choice(tx_types)),
    }


def detect_fraud_rules(
    transaction: dict[str, Any],
    profile: UserProfile,
    recent_transactions: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Rule-based anomaly checks.
    """
    flags: list[str] = []
    amt = float(transaction["transaction_amount"])
    hour = int(transaction["transaction_time"].hour)
    loc = str(transaction["location"])
    device = str(transaction["device_id"])
    start_h, end_h = profile.usual_transaction_hours

    if amt > 2.5 * profile.average_transaction_amount:
        flags.append("HIGH_AMOUNT")
    if loc != profile.home_location:
        flags.append("NEW_LOCATION")
    if not (start_h <= hour <= end_h):
        flags.append("ODD_HOUR")
    if device != profile.primary_device:
        flags.append("NEW_DEVICE")

    # Rapid frequency: >=3 transactions within last 2 minutes.
    tx_time = transaction["transaction_time"]
    recent_window = [t for t in recent_transactions if (tx_time - t["transaction_time"]).total_seconds() <= 120]
    if len(recent_window) >= 2:
        flags.append("RAPID_FREQUENCY")

    rule_score = min(1.0, len(flags) / 5.0)
    return {"rule_flags": flags, "rule_score": rule_score}


def _simulate_training_data(n_rows: int = 3500, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    locations = ["Kampala", "Nairobi", "London", "Dubai", "Johannesburg"]
    tx_types = ["ATM", "POS", "Online"]
    records: list[dict[str, Any]] = []

    for i in range(n_rows):
        scenario = rng.choice(["normal", "suspicious", "fraud_attack"], p=[0.68, 0.2, 0.12])
        home = str(rng.choice(locations))
        avg = float(rng.integers(50_000, 350_000))
        start = int(rng.integers(6, 11))
        end = int(rng.integers(18, 23))
        primary_dev = f"DEV_{int(rng.integers(1_000, 9_999))}"
        profile = UserProfile(
            account_id=f"ACCT_{i%250:05d}",
            home_location=home,
            average_transaction_amount=avg,
            usual_transaction_hours=(start, end),
            primary_device=primary_dev,
        )
        tx_rng = np.random.default_rng(seed + i + 17)
        tx = generate_transaction(str(scenario), profile, tx_rng, [])
        tx_hour = int(tx["transaction_time"].hour)

        amount_ratio = float(tx["transaction_amount"] / max(profile.average_transaction_amount, 1.0))
        is_new_location = int(tx["location"] != profile.home_location)
        is_odd_hour = int(not (start <= tx_hour <= end))
        is_new_device = int(tx["device_id"] != profile.primary_device)
        rapid_tx = int(scenario in ("suspicious", "fraud_attack") and rng.random() < 0.4)
        # Weakly noisy target
        latent = (
            1.35 * (amount_ratio > 2.5)
            + 0.9 * is_new_location
            + 0.8 * is_odd_hour
            + 0.7 * is_new_device
            + 1.0 * rapid_tx
            + 0.45 * (scenario == "fraud_attack")
            - 1.0
        )
        fraud_prob = 1 / (1 + np.exp(-latent))
        y = int(rng.random() < fraud_prob)

        records.append(
            {
                "transaction_amount": tx["transaction_amount"],
                "tx_hour": tx_hour,
                "location": tx["location"],
                "transaction_type": tx["transaction_type"],
                "amount_ratio": amount_ratio,
                "is_new_location": is_new_location,
                "is_odd_hour": is_odd_hour,
                "is_new_device": is_new_device,
                "rapid_tx": rapid_tx,
                "is_fraud": y,
            }
        )
    return pd.DataFrame(records)


def train_ml_model(seed: int = RANDOM_STATE) -> Pipeline:
    """
    Train a simple fraud classifier (Logistic Regression).
    """
    df = _simulate_training_data(seed=seed)
    y = df["is_fraud"].astype(int)
    X = df.drop(columns=["is_fraud"])

    num_cols = [
        "transaction_amount",
        "tx_hour",
        "amount_ratio",
        "is_new_location",
        "is_odd_hour",
        "is_new_device",
        "rapid_tx",
    ]
    cat_cols = ["location", "transaction_type"]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    model = Pipeline(
        steps=[
            ("prep", pre),
            ("clf", LogisticRegression(max_iter=1500, random_state=seed)),
        ]
    )
    model.fit(X, y)
    return model


def calculate_risk_score(
    ml_probability: float,
    rule_score: float,
    transaction_amount: float,
    avg_amount: float,
) -> float:
    """
    0-100 risk score from ML, rules, and amount pressure.
    """
    p = float(np.clip(ml_probability, 0.0, 1.0))
    r = float(np.clip(rule_score, 0.0, 1.0))
    amount_term = float(np.clip(np.log1p(transaction_amount) / np.log1p(max(avg_amount * 5.0, 1.0)), 0.0, 1.0))
    score = 100.0 * (0.70 * p + 0.20 * r + 0.10 * amount_term)
    return float(np.clip(score, 0.0, 100.0))


def make_decision(risk_score: float) -> str:
    """
    0-40 APPROVE, 41-70 FLAG, 71-100 BLOCK.
    """
    if risk_score <= 40:
        return "APPROVE"
    if risk_score <= 70:
        return "FLAG FOR REVIEW"
    return "BLOCK TRANSACTION"


def process_transaction(
    transaction: dict[str, Any],
    profile: UserProfile,
    model: Pipeline,
    recent_transactions: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Full processing pipeline for one transaction.
    """
    # 1) validation
    required = [
        "account_id",
        "transaction_amount",
        "transaction_time",
        "location",
        "device_id",
        "transaction_type",
    ]
    missing = [k for k in required if k not in transaction]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # 2) rule detection
    rule_out = detect_fraud_rules(transaction, profile, recent_transactions)

    # 3) model features
    tx_hour = int(transaction["transaction_time"].hour)
    amount_ratio = float(transaction["transaction_amount"] / max(profile.average_transaction_amount, 1.0))
    features = pd.DataFrame(
        [
            {
                "transaction_amount": float(transaction["transaction_amount"]),
                "tx_hour": tx_hour,
                "location": str(transaction["location"]),
                "transaction_type": str(transaction["transaction_type"]),
                "amount_ratio": amount_ratio,
                "is_new_location": int(transaction["location"] != profile.home_location),
                "is_odd_hour": int(not (profile.usual_transaction_hours[0] <= tx_hour <= profile.usual_transaction_hours[1])),
                "is_new_device": int(transaction["device_id"] != profile.primary_device),
                "rapid_tx": int("RAPID_FREQUENCY" in rule_out["rule_flags"]),
            }
        ]
    )
    fraud_prob = float(model.predict_proba(features)[0][1])
    pred = int(model.predict(features)[0])

    # 4) risk score + decision
    risk = calculate_risk_score(
        fraud_prob,
        float(rule_out["rule_score"]),
        float(transaction["transaction_amount"]),
        profile.average_transaction_amount,
    )
    decision = make_decision(risk)

    # 5) alerts
    alerts: list[str] = []
    if decision == "FLAG FOR REVIEW":
        alerts.extend(
            [
                "⚠️ Suspicious transaction detected",
                "🔔 User notified",
            ]
        )
    elif decision == "BLOCK TRANSACTION":
        alerts.extend(
            [
                "🚫 Transaction blocked",
                "⚠️ Suspicious transaction detected",
                "🔔 User notified",
            ]
        )
    else:
        alerts.append("✅ Transaction approved")

    return {
        "timestamp": datetime.now(),
        **transaction,
        "ml_fraud_probability": round(fraud_prob, 4),
        "ml_prediction": pred,
        "rule_flags": rule_out["rule_flags"],
        "rule_score": round(float(rule_out["rule_score"]), 3),
        "risk_score": round(risk, 2),
        "decision": decision,
        "alerts": alerts,
    }


def run_simulation(
    scenario: str,
    account_id: str,
    n_transactions: int,
    delay_seconds: float,
    show_plot: bool = True,
) -> pd.DataFrame:
    """
    Live simulation loop.
    """
    rng = np.random.default_rng(RANDOM_STATE + len(account_id) + n_transactions)
    profile = get_user_profile(account_id)
    model = train_ml_model()
    logs: list[dict[str, Any]] = []
    recent_tx: list[dict[str, Any]] = []

    print("\n" + "=" * 88)
    print("AI-Based Banking Transaction Simulator with Real-Time Fraud Detection")
    print("=" * 88)
    print(f"Scenario: {scenario}")
    print(f"Account: {account_id}")
    print(f"User profile: home={profile.home_location}, avg_amount={profile.average_transaction_amount:.0f}, "
          f"usual_hours={profile.usual_transaction_hours}, primary_device={profile.primary_device}")
    print("-" * 88)

    for i in range(n_transactions):
        tx = generate_transaction(scenario, profile, rng, recent_tx)
        result = process_transaction(tx, profile, model, recent_tx)
        logs.append(result)
        recent_tx.append(tx)
        if len(recent_tx) > 100:
            recent_tx = recent_tx[-100:]

        # Clean readable output
        print(f"TX #{i+1:02d} | {result['transaction_time']} | {result['transaction_type']} | "
              f"{result['location']} | amt={result['transaction_amount']:.2f}")
        print(f"        ML_p={result['ml_fraud_probability']:.3f} | rules={result['rule_flags']} | "
              f"risk={result['risk_score']:.1f} | decision={result['decision']}")
        for msg in result["alerts"]:
            print(f"        {msg}")
        print("-" * 88)

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    log_df = pd.DataFrame(logs)
    print("\nSimulation complete.")
    print(f"Transactions analyzed: {len(log_df)}")
    print("Decision counts:")
    print(log_df["decision"].value_counts().to_string())

    if show_plot and not log_df.empty:
        plt.figure(figsize=(9, 4))
        plt.plot(log_df.index + 1, log_df["risk_score"], marker="o", linewidth=1.6)
        plt.axhline(40, linestyle="--", linewidth=1, color="#2e8b57", label="Approve/Review")
        plt.axhline(70, linestyle="--", linewidth=1, color="#cc3b3b", label="Review/Block")
        plt.title("Risk Score Trend Over Simulated Transactions")
        plt.xlabel("Transaction Number")
        plt.ylabel("Risk Score")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return log_df


def _ask_scenario() -> str:
    options = {"1": "normal", "2": "suspicious", "3": "fraud_attack"}
    print("\nChoose scenario:")
    print("  1) Normal")
    print("  2) Suspicious")
    print("  3) Fraud attack")
    choice = input("Enter choice [1-3]: ").strip()
    return options.get(choice, "normal")


def main() -> int:
    parser = argparse.ArgumentParser(description="Banking Fraud Detection Real-Time Simulator")
    parser.add_argument("--scenario", choices=["normal", "suspicious", "fraud_attack"], default=None)
    parser.add_argument("--account-id", default="ACCT_00127")
    parser.add_argument("--n-transactions", type=int, default=30)
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between transactions")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    scenario = args.scenario or _ask_scenario()
    run_simulation(
        scenario=scenario,
        account_id=args.account_id,
        n_transactions=max(1, args.n_transactions),
        delay_seconds=max(0.0, args.delay),
        show_plot=not args.no_plot,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
