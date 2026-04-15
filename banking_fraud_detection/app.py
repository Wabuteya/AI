#!/usr/bin/env python3
"""
Institutional bank-style Streamlit fraud intelligence console.

Run from the project directory:
  streamlit run app.py
"""

from __future__ import annotations

APP_ORG = "PARALLAX Bank Fraud Lab"
APP_PAGE_TITLE = "PARALLAX Bank · Fraud Intelligence"
APP_TAGLINE = (
    "Enterprise-style transaction monitoring with hybrid risk scoring, graph context, and analyst-in-the-loop routing."
)

import os
from io import BytesIO

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from realtime_simulator import run_simulation
from fraud_platform.pipeline import (
    SIMULATED_LOCATIONS,
    TRANSACTION_AMOUNT_CURRENCY,
    apply_analyst_feedback,
    anomaly_score_normalized,
    detect_account_fraud,
    evaluate_model,
    load_data,
    monitor_transaction_realtime,
    preprocess_data,
    risk_score,
    split_train_test,
    train_model,
    tree_feature_importance_named,
)


def inject_ui_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f3f3f1;
            color: #262f34;
        }
        section[data-testid="stSidebar"] {
            background: #efefec;
            border-right: 1px solid #d8d8d5;
        }
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] .stMarkdown div {
            color: #3c464c;
        }
        [data-testid="stMetric"] {
            background: #ecece8;
            border: 1px solid #dfdfdb;
            border-radius: 12px;
            padding: 0.5rem;
        }
        /* Make tabs and buttons clearly visible */
        [data-testid="stTabs"] [role="tablist"] {
            gap: 0.3rem;
            border-bottom: 1px solid #cfd6dc;
            padding-bottom: 0.2rem;
        }
        [data-testid="stTabs"] [role="tab"] {
            background: #e8ecef;
            color: #24313a;
            border: 1px solid #c7d0d8;
            border-bottom: none;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            padding: 0.45rem 0.8rem;
            font-weight: 600;
        }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
            background: #1f4d77;
            color: #ffffff;
            border-color: #1f4d77;
        }
        [data-testid="stTabs"] [role="tab"]:hover {
            background: #d9e2ea;
            color: #1d2a33;
        }
        .stButton > button {
            background: #1f4d77;
            color: #ffffff;
            border: 1px solid #174061;
            border-radius: 8px;
            font-weight: 600;
        }
        .stButton > button:hover {
            background: #265b8a;
            border-color: #1f4d77;
            color: #ffffff;
        }
        .stButton > button:disabled {
            background: #d7dde2;
            color: #6d7882;
            border-color: #c6ced5;
        }
        .header-card {
            background: #efefec;
            border: 1px solid #ddddda;
            border-radius: 12px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.8rem;
        }
        .header-card h3 {
            margin: 0;
            color: #1f2a31;
            font-size: 1.9rem;
        }
        .header-card p {
            margin: 0.15rem 0 0 0;
            color: #56656f;
        }
        .section-note {
            color: #6b7780;
            font-size: 0.95rem;
        }
        .panel {
            background: #f7f7f5;
            border: 1px solid #ddddda;
            border-radius: 14px;
            padding: 0.9rem 1rem 1rem 1rem;
            margin-bottom: 0.8rem;
        }
        .panel h4 {
            margin: 0 0 0.5rem 0;
            letter-spacing: 0.03em;
            color: #34424a;
        }
        .welcome-hero {
            background: linear-gradient(120deg, #12385c 0%, #1e4f7d 55%, #2d648f 100%);
            color: #f7fbff;
            border-radius: 16px;
            padding: 1.15rem 1.3rem;
            margin-bottom: 0.9rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .welcome-hero h3 {
            margin: 0;
            font-size: 1.35rem;
        }
        .welcome-hero p {
            margin: 0.35rem 0 0 0;
            color: #dce9f5;
        }
        .welcome-card {
            background: #f7f7f5;
            border: 1px solid #ddddda;
            border-radius: 14px;
            padding: 0.95rem 1rem;
            min-height: 180px;
        }
        .welcome-card h4 {
            margin: 0 0 0.45rem 0;
            color: #2f3f49;
            letter-spacing: 0.02em;
        }
        .welcome-card ul {
            margin: 0.15rem 0 0 1rem;
            color: #55646d;
        }
        .welcome-chip {
            display: inline-block;
            margin: 0.2rem 0.35rem 0 0;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            background: #e8eef5;
            border: 1px solid #d2dce7;
            color: #365067;
            font-size: 0.8rem;
        }
        .decision-ok {
            background: #e9f3e7;
            border-left: 4px solid #2e8b57;
            border-radius: 8px;
            padding: 0.7rem 0.9rem;
        }
        .decision-review {
            background: #fbf3e6;
            border-left: 4px solid #ce8a2a;
            border-radius: 8px;
            padding: 0.7rem 0.9rem;
        }
        .decision-declined {
            background: #f9e8e8;
            border-left: 4px solid #cc3b3b;
            border-radius: 8px;
            padding: 0.7rem 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def decision_banner(decision: str) -> str:
    decision_text = decision.replace("_", " ").title()
    if decision == "approved":
        return f"<div class='decision-ok'><strong>Decision:</strong> {decision_text}</div>"
    if decision == "declined":
        return f"<div class='decision-declined'><strong>Decision:</strong> {decision_text}</div>"
    return f"<div class='decision-review'><strong>Decision:</strong> {decision_text}</div>"


def ensure_output_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def fig_confusion_matrix(cm: np.ndarray, title: str) -> plt.Figure:
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
    return fig


def fig_feature_importance(imp: pd.Series, title: str, top_n: int = 12) -> plt.Figure:
    top = imp.head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    top.plot(kind="barh", ax=ax, color="#2c7fb8")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig


def fig_risk_gauge(score: float) -> plt.Figure:
    s = float(np.clip(score, 0.0, 100.0))
    fig, ax = plt.subplots(figsize=(4, 2.5), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    segments = [
        (0, 35, "#2e8b57"),
        (35, 70, "#ce8a2a"),
        (70, 100, "#cc3b3b"),
    ]
    for low, high, color in segments:
        start = np.pi * (1 - low / 100.0)
        width = np.pi * (high - low) / 100.0
        ax.bar(start, 0.35, width=width, bottom=0.45, color=color, align="edge", alpha=0.9)

    theta = np.pi * (1 - s / 100.0)
    ax.plot([theta, theta], [0.0, 0.72], color="#f8fbff", linewidth=3)
    ax.scatter([theta], [0.0], s=35, color="#f8fbff", zorder=3)
    ax.text(np.pi / 2, -0.18, f"Hybrid Risk: {s:.1f}", ha="center", va="center", fontsize=11, fontweight="bold")
    fig.patch.set_alpha(0.0)
    return fig


def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def score_bar(label: str, value: float, color: str = "#5d9228") -> None:
    safe = float(np.clip(value, 0.0, 1.0))
    c1, c2 = st.columns([5, 1])
    c1.write(label)
    c2.write(f"{value:.3f}")
    st.progress(safe)


def risk_band_decision(risk_score: float, review_cutoff: float, block_cutoff: float) -> tuple[str, str]:
    """
    Simple rubric-style action layer for demos:
    - risk < review_cutoff: Approve
    - review_cutoff <= risk < block_cutoff: Flag for Review
    - risk >= block_cutoff: Block
    """
    r = float(risk_score)
    if r < review_cutoff:
        return "Approved", "✅ Transaction approved (low risk)"
    if r < block_cutoff:
        return "Flagged", "⚠️ Suspicious transaction detected — send for review"
    return "Blocked", "🚫 Transaction blocked due to high risk"


def run_training(n_samples: int, random_state: int, out_dir: str) -> None:
    """Fit models and store artifacts in session_state."""
    out_dir = ensure_output_dir(out_dir)
    csv_path = os.path.join(out_dir, "transactions_simulated.csv")
    df = load_data(n_samples=n_samples, random_state=random_state, save_path=csv_path)
    X, y, preprocessor, num_cols, cat_cols = preprocess_data(df)
    feature_columns = num_cols + cat_cols
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    models = train_model(X_train, y_train, preprocessor)
    lr_model = models["logistic_regression"]
    tree_model = models["decision_tree"]
    ano_model = models["anomaly_detector"]
    ano_calib = models["anomaly_calib"]
    routing_thresholds = models["routing_thresholds"]
    lr_metrics = evaluate_model(lr_model, X_test, y_test, model_name="Logistic Regression", verbose=False)
    tree_metrics = evaluate_model(tree_model, X_test, y_test, model_name="Decision Tree", verbose=False)
    imp = tree_feature_importance_named(tree_model)

    fig_lr = fig_confusion_matrix(lr_metrics["confusion_matrix"], "Confusion Matrix — Logistic Regression")
    fig_imp = fig_feature_importance(imp, "Decision Tree — Top Feature Importances")
    lr_cm_path = os.path.join(out_dir, "confusion_matrix_lr.png")
    imp_path = os.path.join(out_dir, "feature_importance_tree.png")
    fig_lr.savefig(lr_cm_path, dpi=150)
    fig_imp.savefig(imp_path, dpi=150)

    st.session_state.update(
        {
            "trained": True,
            "df": df,
            "csv_path": csv_path,
            "feature_columns": feature_columns,
            "lr_model": lr_model,
            "tree_model": tree_model,
            "anomaly_detector": ano_model,
            "anomaly_calib": ano_calib,
            "routing_thresholds": routing_thresholds,
            "lr_metrics": lr_metrics,
            "tree_metrics": tree_metrics,
            "importances": imp,
            "X_test": X_test,
            "y_test": y_test,
            "out_dir": out_dir,
            "fig_lr": fig_lr,
            "fig_imp": fig_imp,
        }
    )


def main() -> None:
    st.set_page_config(page_title=APP_PAGE_TITLE, layout="wide")
    inject_ui_styles()

    with st.sidebar:
        st.markdown("PARALLAX BANK")
        st.markdown("## Fraud Risk Lab")
        st.caption("Institutional monitoring")
        st.divider()
        st.markdown("### DATA & MODEL OPS")
        out_dir = st.text_input("Output folder", value="outputs")
        n_samples = st.number_input("Simulated rows", min_value=500, max_value=50_000, value=5000, step=500)
        seed = st.number_input("Random seed", min_value=0, value=42, step=1)
        if st.button("Initialize & train models", type="primary", use_container_width=True):
            with st.spinner("Training…"):
                run_training(int(n_samples), int(seed), out_dir.strip() or "outputs")
            st.success("Training complete.")

    if not st.session_state.get("trained"):
        st.markdown(
            """
            <div class="welcome-hero">
              <h3>Welcome to PARALLAX Fraud Command Center</h3>
              <p>Build a full AI fraud workflow in minutes: generate data, train hybrid models, score transactions, and monitor account risk.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(
                """
                <div class="welcome-card">
                  <h4>Quick Start</h4>
                  <ul>
                    <li>Set rows and random seed in the sidebar</li>
                    <li>Click <strong>Initialize & train models</strong></li>
                    <li>Open Decisioning Console to test transactions</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown(
                """
                <div class="welcome-card">
                  <h4>What You Get</h4>
                  <ul>
                    <li>Fraud probability + anomaly scoring</li>
                    <li>Risk score (0-100) and action routing</li>
                    <li>Account-level watchlist detection</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_c:
            st.markdown(
                """
                <div class="welcome-card">
                  <h4>Demo Policies</h4>
                  <ul>
                    <li>Approve / Review / Block bands</li>
                    <li>Analyst feedback threshold tuning</li>
                    <li>Case priority for risky events</li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            """
            <span class="welcome-chip">Hybrid ML</span>
            <span class="welcome-chip">Graph Context</span>
            <span class="welcome-chip">Behavior Signals</span>
            <span class="welcome-chip">Real-Time Alerts</span>
            <span class="welcome-chip">Account Monitoring</span>
            """,
            unsafe_allow_html=True,
        )
        return

    df: pd.DataFrame = st.session_state["df"]
    feature_columns: list[str] = st.session_state["feature_columns"]
    lr_model = st.session_state["lr_model"]
    lr_m = st.session_state["lr_metrics"]
    tree_m = st.session_state["tree_metrics"]
    X_test: pd.DataFrame = st.session_state["X_test"]
    if "blocked_accounts" not in st.session_state:
        st.session_state["blocked_accounts"] = set()

    rt = st.session_state["routing_thresholds"]
    sample = X_test.iloc[:3].copy()
    probs = lr_model.predict_proba(sample)[:, 1]
    ano_model = st.session_state["anomaly_detector"]
    ano_calib = st.session_state["anomaly_calib"]
    ano_n = anomaly_score_normalized(ano_model, ano_calib, sample)
    rows = []
    for i, (_, row) in enumerate(sample.iterrows()):
        rs = risk_score(probs[i], float(row["transaction_amount"]), ano_n[i])
        rows.append(
            {
                "row": i,
                "P(fraud)": round(float(probs[i]), 4),
                "anomaly": round(float(ano_n[i]), 4),
                f"amount ({TRANSACTION_AMOUNT_CURRENCY})": float(row["transaction_amount"]),
                "hybrid_risk": round(rs, 2),
            }
        )

    with st.sidebar:
        st.divider()
        st.markdown("### THRESHOLDS")
        review_t = st.slider(
            "Review threshold",
            min_value=0.10,
            max_value=0.90,
            value=float(rt["review_probability"]),
            step=0.01,
        )
        decline_t = st.slider(
            "Decline threshold",
            min_value=0.20,
            max_value=0.98,
            value=max(float(rt["decline_probability"]), review_t + 0.05),
            step=0.01,
        )
        st.session_state["routing_thresholds"] = {
            "review_probability": float(review_t),
            "decline_probability": float(max(decline_t, review_t + 0.05)),
        }
        st.markdown("#### RISK BAND POLICY")
        risk_review = st.slider(
            "Review cutoff (risk score)",
            min_value=10,
            max_value=80,
            value=40,
            step=1,
        )
        risk_block = st.slider(
            "Block cutoff (risk score)",
            min_value=30,
            max_value=95,
            value=max(70, risk_review + 5),
            step=1,
        )
        st.session_state["risk_band_thresholds"] = {
            "review_cutoff": int(risk_review),
            "block_cutoff": int(max(risk_block, risk_review + 5)),
        }
        st.divider()
        st.markdown("### ACCOUNT CONTROLS")
        blocked_accounts = sorted(list(st.session_state["blocked_accounts"]))
        st.metric("Blocked accounts", len(blocked_accounts))
        if blocked_accounts:
            st.caption(", ".join(blocked_accounts[:6]) + (" ..." if len(blocked_accounts) > 6 else ""))
        else:
            st.caption("No blocked accounts")
        st.divider()
        st.caption(f"Models active · {df.shape[0]:,} rows")

    st.markdown(
        """
        <div class="header-card">
          <h3>Fraud Risk Command Center</h3>
          <p>Hybrid risk scoring · graph context · analyst routing</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total transactions", f"{df.shape[0]:,}", "simulation set")
    k2.metric("Fraud rate", f"{df['is_fraud'].mean():.1%}", "+2.1% vs baseline")
    k3.metric("LR recall", f"{lr_m['recall']:.3f}", "+0.05 vs last run")
    k4.metric("Decline threshold", f"{st.session_state['routing_thresholds']['decline_probability']:.2f}", "cost-calibrated")

    tab_overview, tab_scoring, tab_accounts, tab_live, tab_data = st.tabs(
        [
            "Executive overview",
            "Decisioning console",
            "Account monitoring",
            "Live simulation",
            "Data explorer",
        ]
    )

    with tab_overview:
        st.markdown("<p class='section-note'>Control effectiveness and routing policy behavior on holdout data.</p>", unsafe_allow_html=True)
        probs_holdout = lr_model.predict_proba(X_test)[:, 1]
        review_threshold = st.session_state["routing_thresholds"]["review_probability"]
        decline_threshold = st.session_state["routing_thresholds"]["decline_probability"]
        approved_count = int(np.sum(probs_holdout < review_threshold))
        declined_count = int(np.sum(probs_holdout >= decline_threshold))
        review_count = int(len(probs_holdout) - approved_count - declined_count)

        left, right = st.columns(2)
        with left:
            st.markdown("<div class='panel'><h4>MODEL PERFORMANCE</h4></div>", unsafe_allow_html=True)
            score_bar("LR accuracy", float(lr_m["accuracy"]))
            score_bar("LR precision", float(lr_m["precision"]))
            score_bar("Tree accuracy", float(tree_m["accuracy"]))
            score_bar("Tree recall", float(tree_m["recall"]))
            st.markdown("<div class='panel'><h4>TOP RISK FEATURES</h4></div>", unsafe_allow_html=True)
            top_features = st.session_state["importances"].head(5)
            max_imp = float(max(top_features.max(), 1e-6))
            for fname, imp in top_features.items():
                clean = str(fname).replace("num__", "").replace("cat__", "")
                score_bar(clean.replace("_", " ").title(), float(imp / max_imp))

        with right:
            st.markdown("<div class='panel'><h4>ROUTING POLICY</h4></div>", unsafe_allow_html=True)
            st.write(f"Review threshold: **{review_threshold:.2f}**")
            st.write(f"Decline threshold: **{decline_threshold:.2f}**")
            rb = st.session_state["risk_band_thresholds"]
            st.write(
                "Risk-band logic: "
                f"**< {rb['review_cutoff']} = Approve**, "
                f"**{rb['review_cutoff']}-{rb['block_cutoff'] - 1} = Review**, "
                f"**>= {rb['block_cutoff']} = Block**"
            )
            st.write(f"Auto-approved: **{approved_count:,}**")
            st.write(f"Sent to review: **{review_count:,}**")
            st.write(f"Auto-declined: **{declined_count:,}**")
            st.markdown("<div class='panel'><h4>CONFUSION MATRIX — LOGISTIC REGRESSION</h4></div>", unsafe_allow_html=True)
            st.pyplot(st.session_state["fig_lr"])

    with tab_scoring:
        st.markdown(
            "<p class='section-note'>Decisioning stack: logistic regression + Isolation Forest + policy routing.</p>",
            unsafe_allow_html=True,
        )
        st.session_state.setdefault("decision_console_step", 1)
        step = int(st.session_state["decision_console_step"])
        step = max(1, min(3, step))
        st.session_state["decision_console_step"] = step

        n1, n2, n3, n4 = st.columns([1, 1, 1, 2])
        with n1:
            t1 = "primary" if step == 1 else "secondary"
            if st.button("1 · Input", use_container_width=True, type=t1, key="dec_nav_1"):
                st.session_state["decision_console_step"] = 1
                st.rerun()
        with n2:
            t2 = "primary" if step == 2 else "secondary"
            if st.button("2 · Risk", use_container_width=True, type=t2, key="dec_nav_2"):
                st.session_state["decision_console_step"] = 2
                st.rerun()
        with n3:
            t3 = "primary" if step == 3 else "secondary"
            if st.button("3 · Actions", use_container_width=True, type=t3, key="dec_nav_3"):
                st.session_state["decision_console_step"] = 3
                st.rerun()
        with n4:
            st.caption(f"Step **{step}/3** — use Next/Back below or jump with the buttons above.")

        if step == 1:
            loc_options = list(SIMULATED_LOCATIONS)
            type_options = ["Wire", "Card", "ACH", "ATM", "Online_Payment"]
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                amount = st.number_input(
                    f"transaction_amount ({TRANSACTION_AMOUNT_CURRENCY})",
                    min_value=0.0,
                    value=456_000.0,
                    step=10_000.0,
                    help="All amounts are Ugandan Shillings (UGX).",
                )
                if "transaction_hour_value" not in st.session_state:
                    st.session_state["transaction_hour_value"] = 14
                nav1, nav2, nav3 = st.columns([1, 2, 1])
                with nav1:
                    if st.button("◀", key="hour_prev_btn", help="Previous hour"):
                        st.session_state["transaction_hour_value"] = max(
                            0, int(st.session_state["transaction_hour_value"]) - 1
                        )
                with nav2:
                    hour = st.number_input(
                        "transaction_time (hour)",
                        min_value=0,
                        max_value=23,
                        step=1,
                        key="transaction_hour_value",
                        help="24-hour clock. Use arrows or type directly.",
                    )
                with nav3:
                    if st.button("▶", key="hour_next_btn", help="Next hour"):
                        st.session_state["transaction_hour_value"] = min(
                            23, int(st.session_state["transaction_hour_value"]) + 1
                        )
                        st.rerun()
                hour = int(hour)
                st.progress((hour + 1) / 24.0, text=f"Selected hour: {hour:02d}:00")
            with fc2:
                location = st.selectbox("location", loc_options)
                tx_type = st.selectbox("transaction_type", type_options)
            with fc3:
                account_age = st.number_input("account_age (days)", min_value=1, value=365, step=1)
                prev_fraud = st.selectbox(
                    "previous_fraud_history",
                    [0, 1],
                    format_func=lambda x: "No" if x == 0 else "Yes",
                )

            st.markdown("##### Optional Real-Time Context Signals")
            ctx1, ctx2 = st.columns(2)
            with ctx1:
                payment_instruction_text = st.text_area(
                    "Payment instruction / memo text",
                    value="",
                    help="NLP signal for suspicious phrasing (urgent wire, changed bank details).",
                )
                account_id = st.text_input("account_id", value="ACCT_99999")
                device_id = st.text_input("device_id", value="DEV_99999")
                beneficiary_id = st.text_input("beneficiary_id", value="BEN_99999")
            with ctx2:
                typing_cadence_z = st.slider("typing_cadence_z", 0.0, 5.0, 0.5, 0.1)
                mouse_velocity_z = st.slider("mouse_velocity_z", 0.0, 5.0, 0.5, 0.1)
                session_deviation_z = st.slider("session_deviation_z", 0.0, 5.0, 0.5, 0.1)

            tx = {
                "transaction_amount": float(amount),
                "transaction_time": int(hour),
                "location": location,
                "account_age": int(account_age),
                "transaction_type": tx_type,
                "previous_fraud_history": int(prev_fraud),
                "account_id": account_id.strip(),
                "device_id": device_id.strip(),
                "beneficiary_id": beneficiary_id.strip(),
            }
            st.session_state["last_tx_input"] = tx
            account_is_blocked = tx["account_id"] in st.session_state["blocked_accounts"]
            if account_is_blocked:
                st.error(
                    f"Account `{tx['account_id']}` is currently blocked. "
                    "Transactions from this account are prevented until unblocked."
                )
            if st.button("Run transaction risk check", type="primary"):
                if account_is_blocked:
                    result = {
                        "fraud_prediction": 1,
                        "fraud_probability": 1.0,
                        "risk_score": 100.0,
                        "decision": "declined",
                        "decision_source": "blocked_account",
                        "alerts": [
                            "🚫 Transaction blocked: account is in blocked state.",
                            "Contact fraud operations to review/unblock this account.",
                        ],
                        "reason_codes": ["ACCOUNT_BLOCKED"],
                        "hybrid_layer": False,
                        "graph_risk_score": 0.0,
                        "nlp_risk_score": 0.0,
                        "behavior_risk_score": 0.0,
                        "real_time_monitoring": True,
                        "processing_latency_ms": 0.0,
                        "enterprise_action": "block_transaction",
                        "review_priority": "P1",
                        "analyst_review_required": True,
                        "case_id": f"BLOCK-{tx['account_id']}",
                        "model_lineage": "account-control",
                    }
                else:
                    behavior_profile = {
                        "typing_cadence_z": float(typing_cadence_z),
                        "mouse_velocity_z": float(mouse_velocity_z),
                        "session_deviation_z": float(session_deviation_z),
                    }
                    result = monitor_transaction_realtime(
                        lr_model,
                        tx,
                        feature_columns,
                        anomaly_detector=st.session_state["anomaly_detector"],
                        anomaly_calib=st.session_state["anomaly_calib"],
                        routing_thresholds=st.session_state["routing_thresholds"],
                        recent_transactions=st.session_state["df"].tail(1500),
                        payment_instruction_text=payment_instruction_text,
                        behavior_profile=behavior_profile,
                    )
                st.session_state["last_result"] = result
                st.session_state["decision_console_step"] = 2
                st.rerun()

            b_prev, b_next = st.columns(2)
            with b_prev:
                st.caption("After you run a check, you are taken to **Risk** automatically.")
            with b_next:
                if st.button("Next: Risk & decision →", use_container_width=True, key="dec_next_to_risk"):
                    if st.session_state.get("last_result"):
                        st.session_state["decision_console_step"] = 2
                        st.rerun()
                    else:
                        st.warning("Run a transaction risk check first.")

        result = st.session_state.get("last_result")
        tx_ref = st.session_state.get("last_tx_input", {"account_id": "ACCT_99999"})
        if result:
            rb = st.session_state.get("risk_band_thresholds", {"review_cutoff": 40, "block_cutoff": 70})
            band_status, band_alert = risk_band_decision(
                float(result["risk_score"]),
                float(rb["review_cutoff"]),
                float(rb["block_cutoff"]),
            )
        else:
            band_status, band_alert = "", ""

        if step == 2:
            r_back, r_fwd = st.columns(2)
            with r_back:
                if st.button("← Back to transaction input", use_container_width=True, key="dec_back_to_input"):
                    st.session_state["decision_console_step"] = 1
                    st.rerun()
            with r_fwd:
                if st.button("Next: Actions & alerts →", use_container_width=True, key="dec_next_to_actions"):
                    st.session_state["decision_console_step"] = 3
                    st.rerun()
            if not result:
                st.info("Run a transaction risk check in step 1 to view results.")
            else:
                st.markdown(decision_banner(result["decision"]), unsafe_allow_html=True)
                dcol1, dcol2, dcol3, dcol4 = st.columns(4)
                dcol1.metric("Fraud class", str(result["fraud_prediction"]))
                dcol2.metric("Fraud probability", f"{result['fraud_probability']:.4f}")
                if result.get("hybrid_layer"):
                    dcol3.metric("Anomaly (0-1)", f"{result['anomaly_score_normalized']:.4f}")
                else:
                    dcol3.metric("Anomaly (0-1)", "—")
                dcol4.metric("Hybrid risk", f"{result['risk_score']:.2f}")
                st.write(f"**Rubric status:** `{band_status}`")
                st.write(band_alert)
                cctx1, cctx2, cctx3 = st.columns(3)
                cctx1.metric("Graph risk", f"{result.get('graph_risk_score', 0.0):.3f}")
                cctx2.metric("NLP risk", f"{result.get('nlp_risk_score', 0.0):.3f}")
                cctx3.metric("Behavior risk", f"{result.get('behavior_risk_score', 0.0):.3f}")
                st.caption(f"Processing latency: {result.get('processing_latency_ms', 0.0):.2f} ms")
                ecol1, ecol2, ecol3 = st.columns(3)
                ecol1.metric("Enterprise action", result.get("enterprise_action", "allow_transaction"))
                ecol2.metric("Review priority", result.get("review_priority", "P4"))
                ecol3.metric("Analyst review", "Yes" if result.get("analyst_review_required") else "No")
                if result.get("case_id"):
                    st.write(f"**Case ID:** `{result['case_id']}`")
                st.caption(f"Model lineage: {result.get('model_lineage', 'n/a')}")
                st.pyplot(fig_risk_gauge(float(result["risk_score"])))
                st.write(f"**Decision source:** `{result['decision_source']}`")
                r_back2, r_fwd2 = st.columns(2)
                with r_back2:
                    if st.button("← Back to input", use_container_width=True, key="dec_back_to_input_2"):
                        st.session_state["decision_console_step"] = 1
                        st.rerun()
                with r_fwd2:
                    if st.button("Next: Actions →", use_container_width=True, key="dec_next_to_actions_2"):
                        st.session_state["decision_console_step"] = 3
                        st.rerun()

        elif step == 3:
            a_back, _ = st.columns([1, 2])
            with a_back:
                if st.button("← Back to risk & decision", use_container_width=True, key="dec_back_to_risk"):
                    st.session_state["decision_console_step"] = 2
                    st.rerun()
            if not result:
                st.info("Run a transaction risk check in step 1, then use **Next** to reach actions.")
            else:
                block_candidate = (
                    band_status == "Blocked"
                    or str(result.get("enterprise_action", "")) == "block_transaction"
                    or str(result.get("decision", "")) == "declined"
                )
                st.write("### Account Action Control")
                bc1, bc2 = st.columns(2)
                if block_candidate:
                    if bc1.button("🚫 Block this account", type="primary", key="block_acc_action_tab"):
                        st.session_state["blocked_accounts"].add(str(tx_ref["account_id"]))
                        st.success(f"Account `{tx_ref['account_id']}` blocked from new transactions.")
                else:
                    bc1.button("🚫 Block this account", disabled=True, key="block_acc_action_tab_disabled")
                if bc2.button("✅ Unblock this account", key="unblock_acc_action_tab"):
                    if str(tx_ref["account_id"]) in st.session_state["blocked_accounts"]:
                        st.session_state["blocked_accounts"].remove(str(tx_ref["account_id"]))
                        st.success(f"Account `{tx_ref['account_id']}` unblocked.")
                    else:
                        st.info("This account is not currently blocked.")
                if result.get("reason_codes"):
                    st.write("**Reason codes:** " + ", ".join(result["reason_codes"]))
                if result.get("alerts"):
                    with st.expander("Alerts"):
                        for line in result["alerts"]:
                            st.write(line)
                        st.write(band_alert)
                st.write("### Human Analyst Feedback Loop")
                fb1, fb2 = st.columns(2)
                if fb1.button("Mark as False Positive", key="false_positive_action_tab"):
                    st.session_state["routing_thresholds"] = apply_analyst_feedback(
                        st.session_state["routing_thresholds"], false_positive=True
                    )
                    st.success("Thresholds adjusted to reduce false positives.")
                if fb2.button("Mark as Missed Fraud", key="missed_fraud_action_tab"):
                    st.session_state["routing_thresholds"] = apply_analyst_feedback(
                        st.session_state["routing_thresholds"], missed_fraud=True
                    )
                    st.success("Thresholds adjusted to increase fraud catch rate.")
                result_df = pd.DataFrame([result])
                st.download_button(
                    "Download transaction decision (CSV)",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="transaction_decision.csv",
                    mime="text/csv",
                )

    with tab_data:
        st.subheader("Dataset")
        st.write(f"Shape **{df.shape[0]}** × **{df.shape[1]}**. CSV: `{st.session_state['csv_path']}`")
        st.dataframe(df.head(20), use_container_width=True)
        st.download_button(
            "Download full simulated dataset (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="transactions_simulated.csv",
            mime="text/csv",
        )

    with tab_accounts:
        st.markdown(
            "<p class='section-note'>Detect fraud risk at the individual account level using recent transactions.</p>",
            unsafe_allow_html=True,
        )
        account_ids = sorted(df["account_id"].dropna().astype(str).unique().tolist())
        default_idx = 0 if account_ids else None
        selected_account = st.selectbox(
            "Select account_id",
            account_ids,
            index=default_idx,
        )
        lookback_rows = st.slider(
            "Recent transactions to analyze",
            min_value=5,
            max_value=100,
            value=30,
            step=5,
        )
        if st.button("Run account fraud detection", type="primary"):
            account_df = (
                df[df["account_id"].astype(str) == str(selected_account)]
                .sort_values("event_time")
                .tail(int(lookback_rows))
            )
            account_result = detect_account_fraud(
                lr_model,
                str(selected_account),
                account_df,
                feature_columns,
                anomaly_detector=st.session_state["anomaly_detector"],
                anomaly_calib=st.session_state["anomaly_calib"],
                routing_thresholds=st.session_state["routing_thresholds"],
            )
            st.session_state["last_account_result"] = account_result

        if st.session_state.get("last_account_result"):
            ar = st.session_state["last_account_result"]
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Account", ar["account_id"])
            a2.metric("Risk score", f"{ar['account_risk_score']:.2f}")
            a3.metric("Status", ar["account_status"])
            a4.metric("Transactions scored", ar["transactions_scored"])
            b1, b2, b3 = st.columns(3)
            b1.metric("Mean risk", f"{ar['mean_risk_score']:.2f}")
            b2.metric("Peak risk", f"{ar['peak_risk_score']:.2f}")
            b3.metric("Declines / Step-up", f"{ar['declined_count']} / {ar['step_up_count']}")
            r1, r2 = st.columns(2)
            r1.metric("Review rate", f"{100.0 * float(ar.get('review_rate', 0.0)):.1f}%")
            r2.metric("Decline rate", f"{100.0 * float(ar.get('decline_rate', 0.0)):.1f}%")
            for msg in ar.get("alerts", []):
                st.info(msg)
            ctrl1, ctrl2 = st.columns(2)
            if ar["account_status"] == "high_risk":
                if ctrl1.button("🚫 Block selected account", type="primary"):
                    st.session_state["blocked_accounts"].add(str(ar["account_id"]))
                    st.success(f"Account `{ar['account_id']}` blocked.")
            else:
                ctrl1.button("🚫 Block selected account", disabled=True)
            if ctrl2.button("✅ Unblock selected account"):
                if str(ar["account_id"]) in st.session_state["blocked_accounts"]:
                    st.session_state["blocked_accounts"].remove(str(ar["account_id"]))
                    st.success(f"Account `{ar['account_id']}` unblocked.")
                else:
                    st.info("Selected account is not blocked.")
            top_df = pd.DataFrame(ar.get("top_risky_transactions", []))
            st.subheader("Top risky transactions for this account")
            st.dataframe(top_df, use_container_width=True)
            if not top_df.empty and "risk_score" in top_df.columns:
                st.subheader("Risk trend (recent high-risk transactions)")
                trend_df = top_df.sort_values("event_time")[["event_time", "risk_score"]].copy()
                trend_df["event_time"] = trend_df["event_time"].astype(str)
                st.line_chart(trend_df.set_index("event_time"))

            account_history = (
                df[df["account_id"].astype(str) == str(ar["account_id"])]
                .sort_values("event_time")
                .tail(int(lookback_rows))
            )
            st.subheader("Transaction history view")
            st.dataframe(
                account_history[
                    [
                        "event_time",
                        "transaction_amount",
                        "location",
                        "transaction_type",
                        "previous_fraud_history",
                        "is_fraud",
                    ]
                ],
                use_container_width=True,
            )
            st.download_button(
                "Download account risk result (CSV)",
                data=top_df.to_csv(index=False).encode("utf-8"),
                file_name=f"account_{ar['account_id']}_risk_rows.csv",
                mime="text/csv",
            )

    with tab_live:
        st.markdown(
            "<p class='section-note'>Run end-to-end real-time simulation: "
            "Transaction -> Validation -> Fraud Detection -> Risk -> Decision -> Alert -> Log.</p>",
            unsafe_allow_html=True,
        )
        ls1, ls2, ls3, ls4 = st.columns(4)
        with ls1:
            sim_scenario = st.selectbox(
                "Scenario",
                ["normal", "suspicious", "fraud_attack"],
                index=0,
            )
        with ls2:
            sim_account = st.text_input("Account ID", value="ACCT_00127")
        with ls3:
            sim_n = st.number_input("Transactions", min_value=3, max_value=200, value=20, step=1)
        with ls4:
            sim_delay = st.number_input("Delay (seconds)", min_value=0.0, max_value=3.0, value=0.3, step=0.1)

        if st.button("▶ Start live simulation", type="primary"):
            with st.spinner("Running live simulation..."):
                sim_log = run_simulation(
                    scenario=str(sim_scenario),
                    account_id=str(sim_account).strip() or "ACCT_00127",
                    n_transactions=int(sim_n),
                    delay_seconds=float(sim_delay),
                    show_plot=False,
                    verbose=False,
                )
            st.session_state["sim_log_df"] = sim_log

        if st.session_state.get("sim_log_df") is not None:
            sim_df: pd.DataFrame = st.session_state["sim_log_df"]
            st.success(f"Simulation complete. Processed {len(sim_df)} transactions.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total", len(sim_df))
            m2.metric("Approved", int((sim_df["decision"] == "APPROVE").sum()))
            m3.metric("Flagged", int((sim_df["decision"] == "FLAG FOR REVIEW").sum()))
            m4.metric("Blocked", int((sim_df["decision"] == "BLOCK TRANSACTION").sum()))

            st.subheader("Risk trend")
            trend = sim_df[["timestamp", "risk_score"]].copy()
            trend["timestamp"] = trend["timestamp"].astype(str)
            st.line_chart(trend.set_index("timestamp"))

            st.subheader("Recent simulated transactions")
            show_cols = [
                "timestamp",
                "account_id",
                "transaction_amount",
                "location",
                "transaction_type",
                "ml_fraud_probability",
                "risk_score",
                "decision",
            ]
            st.dataframe(sim_df[show_cols].tail(30), use_container_width=True)
            st.download_button(
                "Download simulation log (CSV)",
                data=sim_df.to_csv(index=False).encode("utf-8"),
                file_name="live_simulation_log.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
