# app.py
"""
AegisBank — Loan Default Prediction  Flask Application

Routes:
  GET  /             → Loan assessment form
  POST /predict      → Run model, save to history, show result
  GET  /dashboard    → Model metrics + confusion matrix
  GET  /history      → All past predictions (filterable)
  GET  /reports      → Individual borrower reports
  GET  /api/metrics  → JSON metrics for dashboard charts
  GET  /api/history  → JSON history for AJAX
  GET  /health       → Healthcheck
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, jsonify, render_template, request, abort

try:
    from .retrain import retrain_model
except ImportError:
    from retrain import retrain_model

# ── project imports ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import MODEL_PATH, FEATURES_PATH, METRICS_PATH, HISTORY_PATH, get_risk_level, PROCESSED_DATA_PATH
from feedback_loop import build_feedback_dataset, update_training_data
from governance import log_decision
from src.shap_explainer import LoanModelExplainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP: load model artefacts
# ─────────────────────────────────────────────────────────────────────────────

def _load_model():
    try:
        m = joblib.load(MODEL_PATH)
        log.info("Model loaded ✅  (%s)", MODEL_PATH)
        return m
    except Exception as e:
        log.error("Model load failed: %s", e)
        return None


def _load_scaler():
    scaler_path = Path(MODEL_PATH).with_name("scaler.pkl")
    try:
        s = joblib.load(scaler_path)
        log.info("Scaler loaded ✅  (%s)", scaler_path)
        return s
    except FileNotFoundError:
        log.info("Scaler not found at %s — using unscaled inputs", scaler_path)
        return None
    except Exception as e:
        log.warning("Scaler load failed: %s — using unscaled inputs", e)
        return None


def _load_features() -> list:
    try:
        with open(FEATURES_PATH, "rb") as f:
            feats = pickle.load(f)
        log.info("Feature list loaded — %d features", len(feats))
        return feats
    except Exception as e:
        log.error("Feature load failed: %s — run utils/preprocessor.py", e)
        return []


def _load_metrics() -> dict:
    defaults = {
        "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
        "f1_score": 0.0, "roc_auc": 0.0,
        "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
    }
    try:
        with open(METRICS_PATH) as f:
            data = json.load(f)
        return {
            "accuracy":  float(data.get("accuracy",  0)),
            "precision": float(data.get("precision", 0)),
            "recall":    float(data.get("recall",    0)),
            "f1_score":  float(data.get("f1_score",  0)),
            "roc_auc":   float(data.get("roc_auc",   0)),
            "confusion_matrix": {
                "tn": int(data.get("confusion_matrix", {}).get("tn", 0)),
                "fp": int(data.get("confusion_matrix", {}).get("fp", 0)),
                "fn": int(data.get("confusion_matrix", {}).get("fn", 0)),
                "tp": int(data.get("confusion_matrix", {}).get("tp", 0)),
            },
        }
    except FileNotFoundError:
        log.warning("model_metrics.json not found — returning zeros. Run evaluate_model.py")
        return defaults
    except Exception as e:
        log.error("Metrics load error: %s", e)
        return defaults


MODEL         = _load_model()
SCALER        = _load_scaler()

def reload_model():
    global MODEL, SCALER
    MODEL = _load_model()
    SCALER = _load_scaler()
    log.info("🔄 Model reloaded after retraining")


MODEL_FEATURES = _load_features()
METRICS       = _load_metrics()

REFERENCE_DATA = pd.read_csv(PROCESSED_DATA_PATH).iloc[:10000]

EXPLAINER = LoanModelExplainer()


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION HISTORY  (JSON file — swap for SQLite in production)
# ─────────────────────────────────────────────────────────────────────────────

def _load_history() -> list:
    try:
        with open(HISTORY_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_history(records: list) -> None:
    Path(HISTORY_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(records, f, indent=2, default=str)


def _append_to_history(record: dict) -> None:
    history = _load_history()
    history.insert(0, record)           # newest first
    history = history[:500]            # cap at 500 entries
    _save_history(history)


def should_retrain():
    history = _load_history()
    return len(history) % 100 == 0 and len(history) != 0


def get_current_data():
    history = _load_history()

    if len(history) < 50:
        return None

    df = pd.DataFrame(history)

    # Extract only numeric features used in drift
    cols = [
        "loan_amnt", "int_rate", "installment", "annual_inc",
        "dti", "fico", "open_acc", "revol_bal", "total_acc"
    ]

    df = df.rename(columns={"fico": "fico_range_low"})
    return df[cols].dropna()


# ─────────────────────────────────────────────────────────────────────────────
# INPUT PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

_NUMERIC_FIELDS = {
    "loan_amnt", "int_rate", "installment", "annual_inc", "dti",
    "fico_range_low", "fico_range_high", "open_acc", "revol_bal",
    "revol_util", "total_acc", "delinq_2yrs", "inq_last_6mths",
    "pub_rec", "pub_rec_bankruptcies", "tax_liens",
    "collections_12_mths_ex_med", "acc_now_delinq", "tot_coll_amt",
    "tot_cur_bal", "avg_cur_bal", "bc_open_to_buy", "bc_util",
    "num_actv_bc_tl", "num_rev_accts", "percent_bc_gt_75",
    # Alternative features for robustness
    "loan_to_income_ratio", "credit_utilization", "fico_avg",
    "mobile_usage_score", "digital_txn_count", "utility_payment_score", "employment_stability",
    "alternative_score",
}

_CATEGORICAL_FIELDS = [
    "term", "grade", "sub_grade", "emp_length",
    "home_ownership", "verification_status", "purpose",
    "addr_state", "initial_list_status", "earliest_cr_line",
]


def create_features_live(df: pd.DataFrame) -> pd.DataFrame:
    df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1e-6)
    df["installment_to_income"] = df["installment"] / (df["annual_inc"] + 1e-6)
    df["credit_utilization"] = df["revol_bal"] / (df["revol_bal"] + 1e-6)
    df["high_dti_flag"] = (df["dti"] > 20).astype(int)
    df["low_fico_flag"] = (df["fico_range_low"] < 600).astype(int)
    return df


def add_economic_features(df):
    df["inflation_rate"] = 0.06
    df["interest_rate_env"] = 0.08
    df["unemployment_rate"] = 0.07

    df["economic_stress"] = (
        df["inflation_rate"] * 0.4 +
        df["unemployment_rate"] * 0.4 +
        df["interest_rate_env"] * 0.2
    )
    return df


def preprocess_input(form_data: dict) -> pd.DataFrame:
    """
    Convert raw form POST data into a 1-row DataFrame aligned to model features.
    """
    if not MODEL_FEATURES:
        raise RuntimeError("Model feature list is empty — run utils/preprocessor.py first.")

    # Fill critical numeric fields when left blank in the form.
    normalized_form_data = dict(form_data)
    normalized_form_data["dti"] = normalized_form_data.get("dti") or 20
    normalized_form_data["revol_util"] = normalized_form_data.get("revol_util") or 50

    row = {feat: 0.0 for feat in MODEL_FEATURES}

    # Numeric fields
    for field in _NUMERIC_FIELDS:
        if field in row:
            try:
                val = float(normalized_form_data.get(field, 0) or 0)
                row[field] = max(val, 0.0)
            except (ValueError, TypeError):
                row[field] = 0.0

    # Handle missing credit users
    if row.get("fico_range_low", 0) == 0:
        # Credit invisible user
        row["alternative_score"] = (
            row.get("mobile_usage_score", 0) * 0.3 +
            row.get("digital_txn_count", 0) * 0.3 +
            row.get("utility_payment_score", 0) * 0.4
        )

    # Categorical → one-hot
    for cat in _CATEGORICAL_FIELDS:
        value = normalized_form_data.get(cat, "")
        if not value:
            continue
        # Naming convention used by pd.get_dummies: "<col>_<value>"
        # Special case: 'term' uses double underscore in some encodings
        candidates = [
            f"{cat}_{value}",
            f"{cat}__{value}",
        ]
        for col_name in candidates:
            if col_name in row:
                row[col_name] = 1.0
                break

    df = pd.DataFrame([row])[MODEL_FEATURES].astype("float32")
    return df


def _validate_input(form_data: dict) -> list:
    """Return a list of validation error strings (empty = valid)."""
    errors = []
    try:
        loan = float(form_data.get("loan_amnt", 0) or 0)
        if loan < 500:
            errors.append("Loan amount must be at least $500.")
    except ValueError:
        errors.append("Loan amount is not a valid number.")

    try:
        inc = float(form_data.get("annual_inc", 0) or 0)
        if inc <= 0:
            errors.append("Annual income must be greater than 0.")
    except ValueError:
        errors.append("Annual income is not a valid number.")

    try:
        fico = float(form_data.get("fico_range_low", 300) or 300)
        if not (300 <= fico <= 850):
            errors.append("FICO score must be between 300 and 850.")
    except ValueError:
        errors.append("FICO score is not a valid number.")

    return errors


def generate_explanation(record):
    return f"""
    Loan Decision Report:
    - Probability of Default: {record['probability']}%
    - Decision: {record['prediction']}
    - Risk Level: {record['risk_level']}
    - Key Factors: {[f['feature'] for f in record['top_features']]}
    """


def calculate_lgd(loan_amount, fico):
    # Simple heuristic
    if fico > 700:
        return 0.2
    elif fico > 600:
        return 0.4
    else:
        return 0.6


def generate_risk_report(record):
    report = f"""
    ===== Loan Risk Report =====
    
    Borrower: {record['borrower']}
    Loan Amount: {record['loan_amnt']}
    
    Probability of Default (PD): {record['probability']}%
    Risk Level: {record['risk_level']}
    
    Decision: {record.get('decision', 'N/A')}
    
    Key Factors:
    """
    
    for f in record.get("explanation", []):
        report += f"\n - {f['feature']}: impact {f['impact']}"
    
    return report


def save_report(report, record_id):
    path = f"reports/{record_id}.txt"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(report)
    return path


def credit_policy(pd):
    if pd > 0.6:
        return "Reject - High Risk"
    elif pd > 0.4:
        return "Manual Review"
    else:
        return "Approve"


def get_risk_category(prob):
    if prob < 0.2:
        return "LOW RISK", False
    elif prob < 0.4:
        return "MEDIUM RISK", True
    elif prob < 0.6:
        return "HIGH RISK", True
    else:
        return "VERY HIGH RISK", True


def get_risk_info(prob):
    if prob < 0.3:
        return "LOW RISK", False
    elif prob < 0.6:
        return "MEDIUM RISK", True
    else:
        return "HIGH RISK", True


def get_decision(prob):
    if prob < 0.3:
        return "LOW RISK", "Repay", False
    elif prob < 0.6:
        return "MEDIUM RISK", "Review", True
    else:
        return "HIGH RISK", "Default", True


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return jsonify({"error": "Model not loaded — run train_model.py first."}), 503

    form_data = request.form.to_dict()

    # Validation
    errors = _validate_input(form_data)
    if errors:
        return render_template("index.html", errors=errors, form_data=form_data)

    try:
        input_df = preprocess_input(form_data)
        input_df = create_features_live(input_df)
        input_df = add_economic_features(input_df)
        # Keep exact same column order as training.
        columns = MODEL_FEATURES
        input_df = input_df.reindex(columns=columns, fill_value=0.0)

        # Explain prediction
        explanation = EXPLAINER.explain_single(input_df)

        # Fairness checks
        fairness_flag = EXPLAINER.check_individual_fairness(form_data)
        bias_flag = EXPLAINER.check_group_bias(form_data)
        sensitive_warning = EXPLAINER.validate_sensitive_features(form_data)

        # Inference using class probability for default risk (PD)
        input_data = input_df
        if SCALER is not None:
            input_data = SCALER.transform(input_data)
            print("Scaler applied: True")
        else:
            print("Scaler applied: False")

        prob = float(MODEL.predict_proba(input_data)[0][1])
        print("Probability:", prob)
        if prob < 0.3:
            print("Warning: prob < 0.3 — possible feature issue")
        probability = prob
        threshold = 0.4

        pd_value = probability
        loan_amount = float(form_data.get("loan_amnt", 0) or 0)
        fico_for_lgd = float(form_data.get("fico_range_low", 0) or 0)
        lgd = calculate_lgd(loan_amount, fico_for_lgd)
        ead = loan_amount
        expected_loss = pd_value * lgd * ead
        expected_profit = loan_amount * (1 - probability) * 0.1 - loan_amount * probability
        income = float(form_data.get("annual_inc", 0) or 0)

        # Risk bands (business-friendly labels)
        if income > 0 and loan_amount > 5 * income:
            risk = "High Risk (Override)"
            verdict = "High Risk (Override)"
            show_warning = True
            print("Override triggered: loan_amount > 5 * annual_inc")
            log.warning("Override triggered for borrower=%s (loan_amount=%.2f, annual_inc=%.2f)",
                        form_data.get("borrower_name", "Anonymous"), loan_amount, income)
        elif prob > 0.6:
            risk = "High Risk"
            verdict = "Default"
            show_warning = True
        elif prob > 0.4:
            risk = "Medium Risk"
            verdict = "Review"
            show_warning = True
        else:
            risk = "Low Risk"
            verdict = "Repay"
            show_warning = False

        prediction = verdict
        decision = verdict
        policy_decision = "Default" if prob > threshold else "Repay"
        risk_label = risk.upper()
        risk_color_map = {
            "LOW RISK": "#22c55e",
            "MEDIUM RISK": "#f59e0b",
            "HIGH RISK": "#f97316",
            "HIGH RISK (OVERRIDE)": "#dc2626",
            "VERY HIGH RISK": "#ef4444",
        }
        if show_warning:
            message = "Default Risk Detected — Review Recommended"
        else:
            message = "Safe Borrower — No Immediate Risk"

        # Check if credit invisible (no FICO score)
        fico = float(form_data.get("fico_range_low", 0) or 0)
        if fico == 0:
            risk_note = "📌 Credit Invisible — evaluated using alternative data"
        else:
            risk_note = "Standard credit evaluation"

        # Build history record
        record = {
            "id":          str(uuid.uuid4()),
            "trace_id":    str(uuid.uuid4()),
            "timestamp":   datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "borrower":    form_data.get("borrower_name", "Anonymous"),
            "loan_amnt":   float(form_data.get("loan_amnt", 0) or 0),
            "int_rate":    float(form_data.get("int_rate", 0) or 0),
            "annual_inc":  float(form_data.get("annual_inc", 0) or 0),
            "fico":        float(form_data.get("fico_range_low", 0) or 0),
            "purpose":     form_data.get("purpose", ""),
            "grade":       form_data.get("grade", ""),
            "prediction":  prediction,
            "verdict":     verdict,
            "decision":    decision,
            "policy_decision": policy_decision,
            "probability": round(probability * 100, 2),
            "PD": round(pd_value, 4),
            "LGD": round(lgd, 2),
            "EAD": round(ead, 2),
            "expected_loss": round(expected_loss, 2),
            "expected_profit": round(expected_profit, 2),
            "model_version": "v1.0",
            "decision_threshold": threshold,
            "features_used": list(input_df.columns),
            "top_features": explanation,
            "fairness_check": fairness_flag,
            "drift_status": "checked",
            "risk_level":  risk_label,
            "show_warning": show_warning,
            "message":     message,
            "color":       risk_color_map.get(risk_label, "#6b7280"),
            "risk_note":   risk_note,
            "raw_input":   form_data,
            "explanation": explanation,
            "fairness": fairness_flag,
            "bias_check": bias_flag,
            "sensitive_warning": sensitive_warning,
        }

        report = generate_risk_report(record)
        report_path = save_report(report, record["id"])
        record["report_path"] = report_path

        _append_to_history(record)
        log_decision(record)

        # Feedback loop
        feedback_data = build_feedback_dataset()

        if feedback_data is not None:
            update_training_data(feedback_data)
            log.info("🔁 Feedback data added to training set")
            retrain_model()
            reload_model()

        from monitoring.drift_detection import detect_drift

        current_data = get_current_data()

        if current_data is not None:
            results, drift_flag = detect_drift(REFERENCE_DATA, current_data)

            if drift_flag:
                log.warning("🚨 DRIFT DETECTED — triggering retraining")

                retrain_model()
                reload_model()

        if should_retrain():
            log.info("⚡ Triggering retraining...")
            retrain_model()
            reload_model()

        return render_template(
            "result.html",
            risk=risk_label,
            show_warning=show_warning,
            prob=prob,
            verdict=verdict,
        )

    except Exception as exc:
        log.exception("Prediction error")
        return render_template("index.html", errors=[f"Prediction failed: {exc}"], form_data=form_data)


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", metrics=METRICS)


@app.route("/history")
def history():
    records = _load_history()
    return render_template("history.html", records=records)


@app.route("/reports")
def reports():
    records = _load_history()
    return render_template("reports.html", records=records)


@app.route("/reports/<record_id>")
def report_detail(record_id: str):
    records = _load_history()
    record  = next((r for r in records if r.get("id") == record_id), None)
    if record is None:
        abort(404)
    return render_template("report_detail.html", record=record)


# ── JSON APIs ────────────────────────────────────────────────────────────────

@app.route("/api/metrics")
def api_metrics():
    return jsonify(METRICS)


@app.route("/api/history")
def api_history():
    q       = request.args.get("q", "").lower()
    records = _load_history()
    if q:
        records = [
            r for r in records
            if q in r.get("borrower", "").lower()
            or q in r.get("purpose", "").lower()
            or q in r.get("risk_level", "").lower()
        ]
    return jsonify(records)


@app.route("/health")
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": MODEL is not None,
        "features":     len(MODEL_FEATURES),
    })


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)


#System Works Like This
#Prediction → Store → Drift Check → If Drift → Retrain → Reload Model ✅