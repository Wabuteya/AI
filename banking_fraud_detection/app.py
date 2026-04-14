#!/usr/bin/env python3
"""
Institutional bank-style Streamlit fraud intelligence console.

Run from the project directory:
  streamlit run app.py
"""

from __future__ import annotations

APP_ORG = "PARALLAX Bank"
APP_PAGE_TITLE = "PARALLAX Bank · Fraud Intelligence"
APP_TAGLINE = (
    "Institutional transaction monitoring with hybrid risk scoring and policy routing."
)

import os
from io import BytesIO

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from fraud_platform.pipeline import (
    SIMULATED_LOCATIONS,
    TRANSACTION_AMOUNT_CURRENCY,
    anomaly_score_normalized,
    evaluate_model,
    load_data,
    preprocess_data,
    predict_transaction,
    risk_score,
    split_train_test,
    train_model,
    tree_feature_importance_named,
)


def inject_ui_styles(dark_mode: bool) -> None:
    page_bg = "#0b1220" if dark_mode else "#f2f4f7"
    card_bg = "#101e33" if dark_mode else "#ffffff"
    text_primary = "#e8edf5" if dark_mode else "#1a2738"
    note_color = "#c2cfdf" if dark_mode else "#30475f"
    decision_ok = "#1f3d2e" if dark_mode else "#eaf7ef"
    decision_review = "#40331d" if dark_mode else "#fff5e8"
    decision_declined = "#442226" if dark_mode else "#ffecec"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {page_bg};
            color: {text_primary};
        }}
        [data-testid="stMetric"] {{
            background: {card_bg};
            border: 1px solid rgba(122, 145, 167, 0.22);
            border-radius: 10px;
            padding: 0.5rem;
        }}
        .hero-card {{
            background: linear-gradient(135deg, #0b1f3a 0%, #15355e 100%);
            color: #f8fbff;
            border-radius: 14px;
            padding: 1rem 1.25rem;
            margin: 0.25rem 0 1rem 0;
            border: 1px solid rgba(255, 255, 255, 0.16);
        }}
        .hero-card p {{
            margin: 0.2rem 0;
        }}
        .section-note {{
            color: {note_color};
            font-size: 0.95rem;
        }}
        .brand-chip {{
            display: inline-block;
            background: rgba(25, 52, 91, 0.18);
            border: 1px solid rgba(98, 124, 158, 0.35);
            color: #dfe9f8;
            border-radius: 999px;
            padding: 0.2rem 0.7rem;
            font-size: 0.8rem;
            letter-spacing: 0.04em;
            margin-bottom: 0.35rem;
        }}
        .decision-ok {{
            background: {decision_ok};
            border-left: 4px solid #2e8b57;
            border-radius: 8px;
            padding: 0.7rem 0.9rem;
        }}
        .decision-review {{
            background: {decision_review};
            border-left: 4px solid #ce8a2a;
            border-radius: 8px;
            padding: 0.7rem 0.9rem;
        }}
        .decision-declined {{
            background: {decision_declined};
            border-left: 4px solid #cc3b3b;
            border-radius: 8px;
            padding: 0.7rem 0.9rem;
        }}
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
    with st.sidebar:
        dark_mode = st.toggle("Dark mode", value=False, help="Toggle dashboard contrast for low-light use.")
    inject_ui_styles(dark_mode)

    logo_col, title_col = st.columns([1, 7])
    with logo_col:
        st.markdown("## 🛡️")
    with title_col:
        st.markdown("<span class='brand-chip'>INSTITUTIONAL FRAUD RISK</span>", unsafe_allow_html=True)
        st.title(APP_ORG)
        st.caption(APP_TAGLINE)
    st.markdown(
        """
        <div class="hero-card">
          <p><strong>Fraud Risk Command Center</strong></p>
          <p>Operate a bank-grade risk workflow: train hybrid models, validate controls, and route decisions by policy thresholds.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown(f"### {APP_ORG}")
        st.caption("Institutional monitoring console")
        st.divider()
        st.header("Data & Model Operations")
        out_dir = st.text_input("Output folder", value="outputs")
        n_samples = st.number_input("Simulated rows", min_value=500, max_value=50_000, value=5000, step=500)
        seed = st.number_input("Random seed", min_value=0, value=42, step=1)
        if st.button("Initialize data & train models", type="primary"):
            with st.spinner("Training…"):
                run_training(int(n_samples), int(seed), out_dir.strip() or "outputs")
            st.success("Training complete.")

    if not st.session_state.get("trained"):
        st.info(
            f"Welcome to **{APP_ORG}** fraud intelligence. Use the sidebar and click "
            "**Initialize data & train models** to begin."
        )
        return

    df: pd.DataFrame = st.session_state["df"]
    feature_columns: list[str] = st.session_state["feature_columns"]
    lr_model = st.session_state["lr_model"]
    lr_m = st.session_state["lr_metrics"]
    tree_m = st.session_state["tree_metrics"]
    X_test: pd.DataFrame = st.session_state["X_test"]

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

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows", f"{df.shape[0]:,}")
    k2.metric("Fraud rate", f"{df['is_fraud'].mean():.1%}")
    k3.metric("LR recall", f"{lr_m['recall']:.3f}")
    k4.metric("Decline threshold", f"{rt['decline_probability']:.3f}")

    tab_overview, tab_scoring, tab_data = st.tabs(["Executive Overview", "Decisioning Console", "Data Explorer"])

    with tab_overview:
        st.markdown("<p class='section-note'>Control effectiveness and routing policy behavior on holdout data.</p>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("LR accuracy", f"{lr_m['accuracy']:.3f}")
        c2.metric("LR precision", f"{lr_m['precision']:.3f}")
        c3.metric("LR recall", f"{lr_m['recall']:.3f}")
        c4.metric("Tree accuracy", f"{tree_m['accuracy']:.3f}")
        c1b, c2b, c3b = st.columns(3)
        c2b.metric("Tree precision", f"{tree_m['precision']:.3f}")
        c3b.metric("Tree recall", f"{tree_m['recall']:.3f}")
        st.caption(
            "Cost-calibrated routing thresholds: "
            f"review >= {rt['review_probability']:.3f}, "
            f"decline >= {rt['decline_probability']:.3f}"
        )
        col_a, col_b = st.columns(2)
        with col_a:
            st.pyplot(st.session_state["fig_lr"])
            st.download_button(
                "Download confusion matrix (PNG)",
                data=fig_to_png_bytes(st.session_state["fig_lr"]),
                file_name="confusion_matrix_lr.png",
                mime="image/png",
            )
        with col_b:
            st.pyplot(st.session_state["fig_imp"])
            st.download_button(
                "Download feature importance (PNG)",
                data=fig_to_png_bytes(st.session_state["fig_imp"]),
                file_name="feature_importance_tree.png",
                mime="image/png",
            )
        st.subheader("Sample hybrid risk scores")
        metrics_df = pd.DataFrame(rows)
        st.table(metrics_df)
        st.download_button(
            "Download sample scores (CSV)",
            data=metrics_df.to_csv(index=False).encode("utf-8"),
            file_name="sample_hybrid_scores.csv",
            mime="text/csv",
        )

    with tab_scoring:
        st.markdown(
            "<p class='section-note'>Decisioning stack: logistic regression + Isolation Forest + policy routing.</p>",
            unsafe_allow_html=True,
        )
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
                    help="24-hour clock. Use arrows or type directly. Transactions at 21-23 are policy-flagged.",
                )
            with nav3:
                if st.button("▶", key="hour_next_btn", help="Next hour"):
                    st.session_state["transaction_hour_value"] = min(
                        23, int(st.session_state["transaction_hour_value"]) + 1
                    )
                    st.rerun()
            hour = int(hour)
            st.progress((hour + 1) / 24.0, text=f"Selected hour: {hour:02d}:00")
            if hour > 20:
                st.warning("Late-hour policy flag will be applied (hour > 20).")
            else:
                st.caption("Normal hour window.")
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

        tx = {
            "transaction_amount": float(amount),
            "transaction_time": int(hour),
            "location": location,
            "account_age": int(account_age),
            "transaction_type": tx_type,
            "previous_fraud_history": int(prev_fraud),
        }
        if st.button("Run transaction risk check", type="primary"):
            result = predict_transaction(
                lr_model,
                tx,
                feature_columns,
                anomaly_detector=st.session_state["anomaly_detector"],
                anomaly_calib=st.session_state["anomaly_calib"],
                routing_thresholds=st.session_state["routing_thresholds"],
            )
            st.session_state["last_result"] = result

        if st.session_state.get("last_result"):
            result = st.session_state["last_result"]
            st.markdown(decision_banner(result["decision"]), unsafe_allow_html=True)
            dcol1, dcol2, dcol3, dcol4 = st.columns(4)
            dcol1.metric("Fraud class", str(result["fraud_prediction"]))
            dcol2.metric("Fraud probability", f"{result['fraud_probability']:.4f}")
            if result.get("hybrid_layer"):
                dcol3.metric("Anomaly (0-1)", f"{result['anomaly_score_normalized']:.4f}")
            else:
                dcol3.metric("Anomaly (0-1)", "—")
            dcol4.metric("Hybrid risk", f"{result['risk_score']:.2f}")
            st.pyplot(fig_risk_gauge(float(result["risk_score"])))
            st.write(f"**Decision source:** `{result['decision_source']}`")
            if result.get("reason_codes"):
                st.write("**Reason codes:** " + ", ".join(result["reason_codes"]))
            if result.get("alerts"):
                with st.expander("Alerts"):
                    for line in result["alerts"]:
                        st.write(line)
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


if __name__ == "__main__":
    main()
