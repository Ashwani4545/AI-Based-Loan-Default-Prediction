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
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, jsonify, render_template, request, abort

# ── project imports ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.config import MODEL_PATH, FEATURES_PATH, METRICS_PATH, HISTORY_PATH, get_risk_level

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
MODEL_FEATURES = _load_features()
METRICS       = _load_metrics()


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
}

_CATEGORICAL_FIELDS = [
    "term", "grade", "sub_grade", "emp_length",
    "home_ownership", "verification_status", "purpose",
    "addr_state", "initial_list_status", "earliest_cr_line",
]


def preprocess_input(form_data: dict) -> pd.DataFrame:
    """
    Convert raw form POST data into a 1-row DataFrame aligned to model features.
    """
    if not MODEL_FEATURES:
        raise RuntimeError("Model feature list is empty — run utils/preprocessor.py first.")

    row = {feat: 0.0 for feat in MODEL_FEATURES}

    # Numeric fields
    for field in _NUMERIC_FIELDS:
        if field in row:
            try:
                val = float(form_data.get(field, 0) or 0)
                row[field] = max(val, 0.0)
            except (ValueError, TypeError):
                row[field] = 0.0

    # Categorical → one-hot
    for cat in _CATEGORICAL_FIELDS:
        value = form_data.get(cat, "")
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

        # Inference via XGBoost DMatrix (avoids sklearn API inconsistencies)
        dmatrix     = xgb.DMatrix(input_df)
        probability = float(MODEL.get_booster().predict(dmatrix)[0])
        prediction  = 1 if probability > 0.50 else 0

        risk = get_risk_level(probability)

        # Build history record
        record = {
            "id":          str(uuid.uuid4()),
            "timestamp":   datetime.utcnow().isoformat() + "Z",
            "borrower":    form_data.get("borrower_name", "Anonymous"),
            "loan_amnt":   float(form_data.get("loan_amnt", 0) or 0),
            "int_rate":    float(form_data.get("int_rate", 0) or 0),
            "annual_inc":  float(form_data.get("annual_inc", 0) or 0),
            "fico":        float(form_data.get("fico_range_low", 0) or 0),
            "purpose":     form_data.get("purpose", ""),
            "grade":       form_data.get("grade", ""),
            "prediction":  prediction,
            "probability": round(probability * 100, 2),
            "risk_level":  risk["label"],
            "color":       risk["color"],
            "raw_input":   form_data,
        }
        _append_to_history(record)

        return render_template(
            "result.html",
            record=record,
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