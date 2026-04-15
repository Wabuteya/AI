#!/usr/bin/env python3
"""
PARALLAX Bank — fraud detection platform (CLI entry point).

Run:
  python main.py

Web app (Streamlit):
  streamlit run app.py

Flow: load data -> preprocess -> train -> evaluate -> plots -> optional interactive transaction.
"""

from __future__ import annotations

import os
import sys

# Headless-safe plotting (servers, CI, some sandboxes) — must run before pyplot import
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraud_platform.pipeline import (
    UGX_PER_USD_DEMO,
    anomaly_score_normalized,
    apply_analyst_feedback,
    evaluate_model,
    load_data,
    monitor_transaction_realtime,
    preprocess_data,
    risk_score,
    split_train_test,
    train_model,
    tree_feature_importance_named,
)


def ensure_output_dir(path: str = "outputs") -> str:
    os.makedirs(path, exist_ok=True)
    return path


def plot_confusion_matrix(cm: np.ndarray, title: str, save_path: str) -> None:
    """Simple heatmap-style confusion matrix (bonus visualization)."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["Pred Legit", "Pred Fraud"],
        yticklabels=["True Legit", "True Fraud"],
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(imp: pd.Series, title: str, save_path: str, top_n: int = 12) -> None:
    """Bar chart of Decision Tree feature importances (bonus)."""
    top = imp.head(top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    top.sort_values().plot(kind="barh", ax=ax, color="#2c7fb8")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def prompt_float(message: str) -> float:
    return float(input(message).strip())


def prompt_int(message: str) -> int:
    return int(input(message).strip())


def prompt_str(message: str) -> str:
    return input(message).strip()


def print_transaction_result(result: dict) -> None:
    print(f"Fraud prediction (0=legit, 1=fraud): {result['fraud_prediction']}")
    print(f"Fraud probability: {result['fraud_probability']:.4f}")
    if result.get("hybrid_layer"):
        print(
            f"Anomaly score (normalized 0-1): {result.get('anomaly_score_normalized', 0):.4f}"
        )
    print(f"Risk score (0-100): {result['risk_score']:.2f}")
    print(f"Decision: {result['decision']} (source: {result['decision_source']})")
    print(
        "Context signals: "
        f"graph={result.get('graph_risk_score', 0.0):.3f}, "
        f"nlp={result.get('nlp_risk_score', 0.0):.3f}, "
        f"behavior={result.get('behavior_risk_score', 0.0):.3f}"
    )
    print(
        "Enterprise action: "
        f"{result.get('enterprise_action', 'allow_transaction')} "
        f"(priority={result.get('review_priority', 'P4')}, "
        f"analyst_review={result.get('analyst_review_required', False)})"
    )
    if result.get("case_id"):
        print(f"Case ID: {result['case_id']}")
    if result.get("real_time_monitoring"):
        print(f"Processing latency: {result.get('processing_latency_ms', 0.0):.3f} ms")
    if result.get("reason_codes"):
        print(f"Reason codes: {', '.join(result['reason_codes'])}")
    for line in result["alerts"]:
        print(line)


def demo_transaction(
    feature_columns: list[str],
    primary_model,
    anomaly_detector,
    anomaly_calib: tuple[float, float],
    routing_thresholds: dict | None = None,
) -> None:
    """Deterministic examples (no stdin) — show decline vs step-up vs approve paths."""
    scenarios = [
        (
            "High-confidence fraud pattern (expect decline or step-up)",
            {
                "transaction_amount": 9200.0 * UGX_PER_USD_DEMO,
                "transaction_time": 2,
                "location": "Foreign",
                "account_age": 14,
                "transaction_type": "Wire",
                "previous_fraud_history": 1,
            },
            "Urgent transfer. Change bank details and do not call.",
            {"typing_cadence_z": 3.0, "mouse_velocity_z": 2.2, "session_deviation_z": 2.8},
        ),
        (
            "Clean small card payment (expect approve)",
            {
                "transaction_amount": 42.0 * UGX_PER_USD_DEMO,
                "transaction_time": 14,
                "location": "NYC",
                "account_age": 800,
                "transaction_type": "Card",
                "previous_fraud_history": 0,
            },
            "",
            {"typing_cadence_z": 0.4, "mouse_velocity_z": 0.5, "session_deviation_z": 0.4},
        ),
        (
            "Policy step-up: new account + moderate amount (expect review, not instant decline)",
            {
                "transaction_amount": 900.0 * UGX_PER_USD_DEMO,
                "transaction_time": 11,
                "location": "Chicago",
                "account_age": 10,
                "transaction_type": "ACH",
                "previous_fraud_history": 0,
            },
            "Invoice payment",
            {"typing_cadence_z": 1.2, "mouse_velocity_z": 1.0, "session_deviation_z": 1.1},
        ),
    ]
    for title, tx, instruction_text, behavior_profile in scenarios:
        print(f"\n=== Demo: {title} ===")
        print(f"Transaction: {tx}")
        result = monitor_transaction_realtime(
            primary_model,
            tx,
            feature_columns,
            anomaly_detector=anomaly_detector,
            anomaly_calib=anomaly_calib,
            routing_thresholds=routing_thresholds,
            recent_transactions=None,
            payment_instruction_text=instruction_text,
            behavior_profile=behavior_profile,
        )
        print("--- Result ---")
        print_transaction_result(result)


def interactive_demo(
    feature_columns: list[str],
    primary_model,
    anomaly_detector,
    anomaly_calib: tuple[float, float],
    routing_thresholds: dict | None = None,
) -> None:
    """Simulate a teller / API client entering one transaction."""
    print("\n=== Simulated new transaction (user input) ===")
    print("Enter transaction fields (transaction_amount is in UGX — Ugandan Shillings).")
    try:
        tx = {
            "transaction_amount": prompt_float("transaction_amount: "),
            "transaction_time": prompt_int("transaction_time (0-23 hour): "),
            "location": prompt_str(
                "location (NYC/LA/Chicago/Miami/Seattle/Online/Foreign/Kampala/Entebbe/Jinja/Mbarara/Gulu): "
            ),
            "account_age": prompt_int("account_age (days): "),
            "transaction_type": prompt_str(
                "transaction_type (Wire/Card/ACH/ATM/Online_Payment): "
            ),
            "previous_fraud_history": prompt_int("previous_fraud_history (0 or 1): "),
        }
    except ValueError:
        print("Invalid input; skipping interactive demo.")
        return

    result = monitor_transaction_realtime(
        primary_model,
        tx,
        feature_columns,
        anomaly_detector=anomaly_detector,
        anomaly_calib=anomaly_calib,
        routing_thresholds=routing_thresholds,
        recent_transactions=None,
        payment_instruction_text=prompt_str("payment_instruction_text (optional): "),
        behavior_profile={
            "typing_cadence_z": prompt_float("typing_cadence_z (0-5, optional baseline deviation): "),
            "mouse_velocity_z": prompt_float("mouse_velocity_z (0-5): "),
            "session_deviation_z": prompt_float("session_deviation_z (0-5): "),
        },
    )
    print("\n--- Result ---")
    print_transaction_result(result)


def main() -> int:
    out_dir = ensure_output_dir()

    # 1) Data
    csv_path = os.path.join(out_dir, "transactions_simulated.csv")
    df = load_data(save_path=csv_path)
    print(f"Dataset shape: {df.shape}. Saved sample CSV to {csv_path}")

    # 2) Preprocess
    X, y, preprocessor, num_cols, cat_cols = preprocess_data(df)
    feature_columns = num_cols + cat_cols

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # 3) Train
    models = train_model(X_train, y_train, preprocessor)
    lr_model = models["logistic_regression"]
    tree_model = models["decision_tree"]
    ano_model = models["anomaly_detector"]
    ano_calib = models["anomaly_calib"]
    routing_thresholds = models["routing_thresholds"]
    print(
        "Routing thresholds (cost-tuned): "
        f"review>={routing_thresholds['review_probability']:.3f}, "
        f"decline>={routing_thresholds['decline_probability']:.3f}"
    )
    routing_thresholds = apply_analyst_feedback(
        routing_thresholds, false_positive=False, missed_fraud=False
    )

    # 4) Evaluate both (assignment asks for metrics + explanation)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, model_name="Logistic Regression")
    _ = evaluate_model(tree_model, X_test, y_test, model_name="Decision Tree")

    # 5) Bonus plots
    plot_confusion_matrix(
        lr_metrics["confusion_matrix"],
        "Confusion Matrix — Logistic Regression",
        os.path.join(out_dir, "confusion_matrix_lr.png"),
    )
    imp = tree_feature_importance_named(tree_model)
    plot_feature_importance(
        imp,
        "Decision Tree — Top Feature Importances",
        os.path.join(out_dir, "feature_importance_tree.png"),
    )
    print(f"\nSaved plots under: {out_dir}/")

    # 6) Example batch scoring (hybrid: LR + Isolation Forest + amount)
    sample = X_test.iloc[:3].copy()
    probs = lr_model.predict_proba(sample)[:, 1]
    ano_n = anomaly_score_normalized(ano_model, ano_calib, sample)
    print("\n--- Sample hybrid risk scores (first 3 test rows) ---")
    for i, (_, row) in enumerate(sample.iterrows()):
        rs = risk_score(probs[i], float(row["transaction_amount"]), ano_n[i])
        print(
            f"Row {i}: P(fraud)={probs[i]:.3f}, anomaly={ano_n[i]:.3f}, "
            f"amount_UGX={row['transaction_amount']:.2f}, hybrid_risk={rs:.2f}"
        )

    # 7) Single-transaction path (no stdin)
    if "--demo-tx" in sys.argv:
        demo_transaction(
            feature_columns,
            lr_model,
            ano_model,
            ano_calib,
            routing_thresholds=routing_thresholds,
        )
    if "--no-interactive" in sys.argv:
        return 0

    # 8) Interactive transaction (default when run as `python main.py`)
    interactive_demo(
        feature_columns,
        lr_model,
        ano_model,
        ano_calib,
        routing_thresholds=routing_thresholds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
