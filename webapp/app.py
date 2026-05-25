# webapp/app.py
"""
AegisBank — Loan Default Prediction Flask Application

Routes:
  GET  /             → Loan assessment form (login required)
  GET  /signin       → Sign-in page
  GET  /signup       → Sign-up page
  GET  /signout      → Sign out
  GET  /dashboard    → Model metrics + confusion matrix
  GET  /history      → All past predictions (filterable)
  GET  /reports      → Individual borrower reports
  GET  /compare      → Side-by-side borrower comparison
  GET  /audit        → Compliance audit log
  GET  /heatmap      → Geographic risk heatmap
  GET  /timeline     → Borrower risk over time
  GET  /batch        → Batch prediction upload
  GET  /admin        → Admin panel
  POST /api/v1/predict  → REST API endpoint
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
import time
import secrets
import warnings
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import (
    Flask, jsonify, render_template, render_template_string, request, abort,
    redirect, url_for, session, flash,
)
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    current_user, login_required as fl_login_required,
)
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from flask_socketio import SocketIO, emit
from flask_swagger_ui import get_swaggerui_blueprint

# Suppress benign XGBoost pickling warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

try:
    from .retrain import retrain_model
except ImportError:
    from retrain import retrain_model

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import (
    CHAMPION_MODEL_PATH, CHALLENGER_MODEL_PATH, MODEL_PATH,
    FEATURES_PATH, METRICS_PATH, CHALLENGER_METRICS_PATH,
    HISTORY_PATH, get_risk_level, PROCESSED_DATA_PATH,
)
from feedback_loop import build_feedback_dataset, update_training_data
from governance import log_decision
from src.shap_explainer import LoanModelExplainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.secret_key = os.environ.get("AEGIS_SECRET_KEY", "aegisbank-dev-secret-key-change-in-prod")
socketio = SocketIO(app, async_mode="threading")

# ── SWAGGER UI ────────────────────────────────────────────────────────────────
SWAGGER_URL = "/api/docs"
API_URL     = "/static/swagger.json"
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL, API_URL,
    config={"app_name": "AegisBank Risk Engine API"},
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# ── DATABASE ──────────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).resolve().parent / "aegisbank.db"
app.config["SQLALCHEMY_DATABASE_URI"]        = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ── FLASK-MAIL ────────────────────────────────────────────────────────────────
app.config["MAIL_SERVER"]         = os.environ.get("MAIL_SERVER",   "smtp.gmail.com")
app.config["MAIL_PORT"]           = int(os.environ.get("MAIL_PORT", "587"))
app.config["MAIL_USE_TLS"]        = True
app.config["MAIL_USERNAME"]       = os.environ.get("MAIL_USERNAME", "")
app.config["MAIL_PASSWORD"]       = os.environ.get("MAIL_PASSWORD", "")
app.config["MAIL_DEFAULT_SENDER"] = os.environ.get("MAIL_USERNAME", "noreply@aegisbank.com")
mail = Mail(app)

# ── FLASK-LOGIN ───────────────────────────────────────────────────────────────
login_manager = LoginManager(app)
login_manager.login_view             = "signin"
login_manager.login_message          = "Please sign in to access this page."
login_manager.login_message_category = "error"


# ── USER MODEL ────────────────────────────────────────────────────────────────
class User(UserMixin, db.Model):
    __tablename__ = "users"

    id             = db.Column(db.Integer,     primary_key=True)
    email          = db.Column(db.String(120), unique=True, nullable=False, index=True)
    first_name     = db.Column(db.String(80),  nullable=False)
    last_name      = db.Column(db.String(80),  nullable=False)
    role           = db.Column(db.String(30),  nullable=False, default="analyst")
    password_hash  = db.Column(db.String(256), nullable=False)
    created_at     = db.Column(db.DateTime,    default=datetime.utcnow)
    is_active      = db.Column(db.Boolean,     default=True,  nullable=False)
    email_verified = db.Column(db.Boolean,     default=False, nullable=False)

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f"<User {self.email} [{self.role}]>"


class ApiKey(db.Model):
    __tablename__ = "api_keys"

    id         = db.Column(db.Integer,     primary_key=True)
    user_id    = db.Column(db.Integer,     db.ForeignKey("users.id"), nullable=False)
    key_hash   = db.Column(db.String(256), unique=True, nullable=False, index=True)
    created_at = db.Column(db.DateTime,    default=datetime.utcnow)
    user       = db.relationship("User",   backref=db.backref("api_keys", lazy=True))

    def set_key(self, raw_key: str):
        self.key_hash = generate_password_hash(raw_key)

    def check_key(self, raw_key: str) -> bool:
        return check_password_hash(self.key_hash, raw_key)


@login_manager.user_loader
def load_user(user_id: str):
    return db.session.get(User, int(user_id))


# ── TOKEN HELPERS ─────────────────────────────────────────────────────────────
_ts = URLSafeTimedSerializer(app.secret_key)


def _generate_token(email: str, salt: str) -> str:
    return _ts.dumps(email, salt=salt)


def _verify_token(token: str, salt: str, max_age: int = 3600):
    try:
        return _ts.loads(token, salt=salt, max_age=max_age)
    except (SignatureExpired, BadSignature):
        return None


def _send_email(to: str, subject: str, html_body: str):
    """Send email; prints to console in dev mode (MAIL_USERNAME not set)."""
    if not app.config["MAIL_USERNAME"]:
        log.info("\n" + "=" * 60)
        log.info("[DEV EMAIL] To: %s | Subject: %s", to, subject)
        log.info(html_body)
        log.info("=" * 60 + "\n")
        return
    try:
        msg = Message(subject=subject, recipients=[to], html=html_body)
        mail.send(msg)
    except Exception as exc:
        log.error("Email send failed: %s", exc)


def _seed_default_users():
    """Create 4 demo accounts if they don't already exist (idempotent)."""
    defaults = [
        ("Admin",      "User",    "admin",        "admin@aegisbank.com",      "Admin@1234"),
        ("Risk",       "Manager", "risk_manager",  "risk@aegisbank.com",       "Risk@1234"),
        ("Credit",     "Analyst", "analyst",       "analyst@aegisbank.com",    "Analyst@1234"),
        ("Compliance", "Officer", "compliance",    "compliance@aegisbank.com", "Comply@1234"),
    ]
    for first, last, role, email, pwd in defaults:
        if not User.query.filter_by(email=email).first():
            u = User(first_name=first, last_name=last, role=role,
                     email=email, email_verified=True)
            u.set_password(pwd)
            db.session.add(u)
    db.session.commit()


# ── AUTH DECORATORS ───────────────────────────────────────────────────────────
def login_required(f):
    """Redirect to /signin if not authenticated."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated:
            flash("Please sign in to access this page.", "error")
            return redirect(url_for("signin"))
        return f(*args, **kwargs)
    return decorated


def role_required(*allowed_roles):
    """Restrict route to specific roles."""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not current_user.is_authenticated:
                flash("Please sign in to access this page.", "error")
                return redirect(url_for("signin"))
            if current_user.role not in allowed_roles:
                flash(f"Access denied. Required role: {' or '.join(allowed_roles)}.", "error")
                return redirect(url_for("index"))
            return f(*args, **kwargs)
        return decorated
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP: load model artefacts
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(path: str):
    try:
        m = joblib.load(path)
        log.info("Model loaded ✅  (%s)", path)
        return m
    except Exception as e:
        log.error("Model load failed from %s: %s", path, e)
        return None


def _load_features() -> list:
    try:
        with open(FEATURES_PATH, "rb") as f:
            feats = pickle.load(f)
        log.info("Feature list loaded — %d features", len(feats))
        return list(feats)
    except Exception as e:
        log.error("Feature load failed: %s — run src/train_model.py first", e)
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
        log.warning("model_metrics.json not found — returning zeros. Run src/evaluate_model.py")
        return defaults
    except Exception as e:
        log.error("Metrics load error: %s", e)
        return defaults


def _load_threshold() -> float:
    """Load the Youden's J optimal threshold saved by train_model.py."""
    try:
        with open(METRICS_PATH) as f:
            data = json.load(f)
        t = data.get("decision_threshold")
        if t is not None:
            return float(t)
    except Exception:
        pass
    return 0.5


# Load artefacts at startup
MODEL            = _load_model(CHAMPION_MODEL_PATH) or _load_model(MODEL_PATH)
CHALLENGER_MODEL = _load_model(CHALLENGER_MODEL_PATH)
MODEL_FEATURES   = _load_features()
METRICS          = _load_metrics()
REFERENCE_DATA   = pd.read_csv(PROCESSED_DATA_PATH).iloc[:10_000]
EXPLAINER        = LoanModelExplainer()


def reload_model() -> None:
    """Reload all artefacts after retraining."""
    global MODEL, CHALLENGER_MODEL, MODEL_FEATURES, METRICS
    MODEL            = _load_model(CHAMPION_MODEL_PATH) or _load_model(MODEL_PATH)
    CHALLENGER_MODEL = _load_model(CHALLENGER_MODEL_PATH)
    MODEL_FEATURES   = _load_features()
    METRICS          = _load_metrics()
    EXPLAINER.reload()
    log.info("🔄 Models + SHAP explainer reloaded after retraining")


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION HISTORY
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
    history.insert(0, record)
    _save_history(history[:500])


def should_retrain() -> bool:
    history = _load_history()
    return len(history) >= 100 and len(history) % 100 == 0


def get_current_data() -> pd.DataFrame | None:
    history = _load_history()
    if len(history) < 50:
        return None

    cols = [
        "loan_amnt", "int_rate", "installment", "annual_inc",
        "dti", "fico_range_low", "open_acc", "revol_bal", "total_acc",
    ]
    rows = []
    for r in history:
        raw = r.get("raw_input", {})
        rows.append({
            "loan_amnt":      float(r.get("loan_amnt")      or raw.get("loan_amnt",      0) or 0),
            "int_rate":       float(r.get("int_rate")        or raw.get("int_rate",       0) or 0),
            "installment":    float(raw.get("installment",   0) or 0),
            "annual_inc":     float(r.get("annual_inc")      or raw.get("annual_inc",     0) or 0),
            "dti":            float(raw.get("dti",           0) or 0),
            "fico_range_low": float(r.get("fico")            or raw.get("fico_range_low", 0) or 0),
            "open_acc":       float(raw.get("open_acc",      0) or 0),
            "revol_bal":      float(raw.get("revol_bal",     0) or 0),
            "total_acc":      float(raw.get("total_acc",     0) or 0),
        })

    df = pd.DataFrame(rows)[cols].dropna()
    return df if not df.empty else None


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
    "mobile_usage_score", "digital_txn_count",
    "utility_payment_score", "employment_stability",
}

_CATEGORICAL_FIELDS = [
    "term", "grade", "sub_grade", "emp_length",
    "home_ownership", "verification_status", "purpose",
    "addr_state", "initial_list_status",
    # earliest_cr_line removed — not collected in form, adding it here
    # caused a mismatch between form data and model features
]


def create_features_live(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute engineered features — MUST exactly mirror create_features()
    in train_model.py. Any divergence causes train/serve skew.
    """
    df["loan_to_income"]        = df["loan_amnt"]   / (df["annual_inc"] + 1e-6)
    df["installment_to_income"] = df["installment"] / (df["annual_inc"] + 1e-6)
    # credit_utilization: revol_bal / (revol_bal + bc_open_to_buy)
    # HEAD version used revol_bal / (revol_bal + 1e-6) — missing bc_open_to_buy
    df["credit_utilization"]    = df["revol_bal"]   / (df["revol_bal"] + df["bc_open_to_buy"] + 1e-6)
    df["payment_capacity"]      = df["annual_inc"]  - df["installment"] * 12
    df["credit_stress"]         = df["dti"]         * df["loan_amnt"]
    df["recent_inquiries_flag"] = (df["inq_last_6mths"] > 3).astype(int)
    df["high_dti_flag"]         = (df["dti"] > 20).astype(int)
    df["low_fico_flag"]         = (df["fico_range_low"] < 600).astype(int)
    return df


def preprocess_input(form_data: dict) -> pd.DataFrame:
    """
    Convert raw form POST data into a 1-row DataFrame aligned to model features.

    Uses actual form fields (annual_inc, fico_range_low) directly.
    The HEAD version fabricated these from monthly_income and credit_history_years
    which caused the audit log to store invented financial data.
    """
    if not MODEL_FEATURES:
        raise RuntimeError("Model feature list is empty — run src/train_model.py first.")

    normalized = dict(form_data)
    # Fill optional fields that the form may leave blank
    normalized["dti"]       = normalized.get("dti")       or 0
    normalized["revol_util"] = normalized.get("revol_util") or 0

    row = {feat: 0.0 for feat in MODEL_FEATURES}

    # Map numeric fields
    for field in _NUMERIC_FIELDS:
        if field in row:
            try:
                val = float(normalized.get(field, 0) or 0)
                row[field] = max(val, 0.0)
            except (ValueError, TypeError):
                row[field] = 0.0

    # Alternative score for credit-invisible users (FICO == 0)
    if row.get("fico_range_low", 0) == 0:
        row["alternative_score"] = (
            row.get("mobile_usage_score",    0) * 0.3 +
            row.get("digital_txn_count",     0) * 0.3 +
            row.get("utility_payment_score", 0) * 0.4
        )

    # Map categorical → one-hot columns
    for cat in _CATEGORICAL_FIELDS:
        value = normalized.get(cat, "")
        if not value:
            continue
        for col_name in (f"{cat}_{value}", f"{cat}__{value}"):
            if col_name in row:
                row[col_name] = 1.0
                break

    df = pd.DataFrame([row])[MODEL_FEATURES].astype("float32")
    return df


def _validate_input(form_data: dict) -> list:
    errors = []
    try:
        if float(form_data.get("loan_amnt", 0) or 0) < 500:
            errors.append("Loan amount must be at least $500.")
    except ValueError:
        errors.append("Loan amount is not a valid number.")
    try:
        if float(form_data.get("annual_inc", 0) or 0) <= 0:
            errors.append("Annual income must be greater than 0.")
    except ValueError:
        errors.append("Annual income is not a valid number.")
    try:
        fico = float(form_data.get("fico_range_low", 300) or 300)
        if not (300 <= fico <= 850):
            errors.append("FICO score must be between 300 and 850.")
    except ValueError:
        errors.append("FICO score is not a valid number.")
    try:
        dti = float(form_data.get("dti", 0) or 0)
        if not (0 <= dti <= 100):
            errors.append("Debt-to-income ratio must be between 0 and 100.")
    except ValueError:
        errors.append("DTI is not a valid number.")
    return errors


# ─────────────────────────────────────────────────────────────────────────────
# FINANCIAL CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_lgd(fico: float) -> float:
    """
    Loss Given Default — estimated from FICO tier.
    loan_amount parameter removed: LGD is a rate (%), not an amount.
    Using 5 tiers instead of 3 for finer granularity.
    """
    if fico >= 750: return 0.15
    if fico >= 700: return 0.25
    if fico >= 650: return 0.35
    if fico >= 600: return 0.45
    return 0.55


def calculate_expected_profit(loan_amount: float, annual_rate_pct: float,
                               pd_value: float, lgd: float) -> float:
    """
    Expected Profit = Revenue if repaid - Expected Loss if defaulted.

    Uses the actual interest rate from the form, not a hardcoded 10%.
    Uses LGD (not full loan amount) for the loss side since partial
    recovery typically occurs on defaulted loans.
    """
    annual_rate = annual_rate_pct / 100.0
    revenue_if_repaid = loan_amount * annual_rate * (1 - pd_value)
    loss_if_default   = loan_amount * pd_value * lgd
    return round(revenue_if_repaid - loss_if_default, 2)


# ─────────────────────────────────────────────────────────────────────────────
# OVERRIDE RULES
# ─────────────────────────────────────────────────────────────────────────────

def check_overrides(form_data: dict) -> tuple[bool, str | None, float | None]:
    """
    Apply hard underwriting rules that override the model probability.

    Returns:
        (override_triggered, reason_string, adjusted_display_probability)
    """
    loan_amount = float(form_data.get("loan_amnt",      0) or 0)
    annual_inc  = float(form_data.get("annual_inc",     0) or 0)
    fico        = float(form_data.get("fico_range_low", 0) or 0)
    dti         = float(form_data.get("dti",            0) or 0)
    delinq      = float(form_data.get("delinq_2yrs",    0) or 0)
    pub_rec     = float(form_data.get("pub_rec",        0) or 0)

    if fico > 0 and fico < 500:
        return True, f"FICO score {int(fico)} is critically low (< 500) — automatic decline", 0.94

    if fico > 0 and fico < 580 and annual_inc > 0 and loan_amount > annual_inc * 0.5:
        return True, (
            f"Sub-prime FICO ({int(fico)}) combined with loan amount "
            f"exceeding 50% of annual income"
        ), 0.82

    if dti > 40:
        adj = min(0.82 + (dti - 40) * 0.005, 0.99)
        return True, f"Debt-to-income ratio {dti:.1f}% exceeds the 40% hard limit", adj

    if delinq >= 3:
        return True, f"{int(delinq)} delinquencies in last 2 years", 0.87

    if pub_rec >= 2:
        return True, f"{int(pub_rec)} public records (bankruptcies/judgements) on file", 0.83

    if annual_inc > 0 and loan_amount > 5 * annual_inc:
        ratio = loan_amount / annual_inc
        adj   = min(0.88 + (ratio - 5) * 0.015, 0.99)
        return True, (
            f"Loan amount (${loan_amount:,.0f}) exceeds 5× annual income "
            f"(${annual_inc:,.0f})"
        ), adj

    return False, None, None


# ─────────────────────────────────────────────────────────────────────────────
# RISK CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

RISK_COLOR_MAP = {
    "LOW RISK":                  "#22c55e",
    "MEDIUM RISK":               "#f59e0b",
    "HIGH RISK":                 "#f97316",
    "VERY HIGH RISK":            "#ef4444",
    "VERY HIGH RISK (OVERRIDE)": "#dc2626",
}


def classify_risk(prob: float, override: bool, fico: float, loan_amount: float,
                  annual_inc: float) -> tuple[str, str, bool]:
    """
    Map probability → (risk_label, verdict, show_warning).
    Applies sub-prime FICO soft tightening when income does not
    comfortably cover the loan.
    """
    if override:
        return "VERY HIGH RISK (OVERRIDE)", "Default", True

    risk_info    = get_risk_level(prob)
    risk_label_v = risk_info["label"]

    # Soft tighten: FICO 500-619 AND loan > 30% of income → bump LOW to MEDIUM
    if (fico > 0 and fico < 620
            and annual_inc > 0
            and loan_amount / annual_inc > 0.30
            and risk_label_v == "LOW RISK"):
        risk_label_v = "MEDIUM RISK"

    if risk_label_v == "LOW RISK":
        return "LOW RISK", "Repay", False
    elif risk_label_v == "MEDIUM RISK":
        return "MEDIUM RISK", "Review", True
    else:
        return risk_label_v, "Default", True


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

_REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def generate_risk_report(record: dict) -> str:
    lines = [
        "===== AegisBank Loan Risk Report =====",
        f"Borrower:    {record.get('borrower', 'Anonymous')}",
        f"Loan Amount: ${record.get('loan_amnt', 0):,.2f}",
        f"Annual Inc:  ${record.get('annual_inc', 0):,.2f}",
        "",
        f"Default Probability: {record.get('probability', 0):.2f}%",
        f"Risk Level:          {record.get('risk_level', 'N/A')}",
        f"Decision:            {record.get('decision', 'N/A')}",
        f"Override:            {record.get('override_triggered', False)}",
        "",
        "Key Risk Drivers (SHAP):",
    ]
    for feat in record.get("top_features", []):
        direction = "↑ risk" if feat.get("increases_risk") else "↓ risk"
        lines.append(
            f"  {feat.get('feature', ''):<35} "
            f"{feat.get('shap_value', feat.get('impact', 0)):+.6f}  ({direction})"
        )
    return "\n".join(lines)


def save_report(report: str, record_id: str) -> str:
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _REPORTS_DIR / f"{record_id}.txt"
    with open(path, "w") as f:
        f.write(report)
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# _score_borrower — used by /compare and /api/v1/predict
# ─────────────────────────────────────────────────────────────────────────────

def _score_borrower(form_data: dict) -> dict:
    """Run the full prediction pipeline on one borrower dict."""
    try:
        input_df = preprocess_input(form_data)
        input_df = create_features_live(input_df)
        input_df = input_df.reindex(columns=MODEL_FEATURES, fill_value=0.0)

        model_prob = float(MODEL.predict_proba(input_df)[0][1])

        loan_amount = float(form_data.get("loan_amnt",      0) or 0)
        annual_inc  = float(form_data.get("annual_inc",     0) or 0)
        fico        = float(form_data.get("fico_range_low", 0) or 0)
        int_rate    = float(form_data.get("int_rate",       0) or 0)

        override, override_reason, adj_prob = check_overrides(form_data)
        display_prob = max(adj_prob or model_prob, model_prob) if override else model_prob

        lgd            = calculate_lgd(fico)
        expected_loss  = display_prob * lgd * loan_amount
        expected_profit = calculate_expected_profit(loan_amount, int_rate, display_prob, lgd)

        risk_label, verdict, show_warning = classify_risk(
            display_prob, override, fico, loan_amount, annual_inc
        )

        # Shadow model (A/B testing)
        challenger_prob = 0.0
        if CHALLENGER_MODEL:
            challenger_prob = float(CHALLENGER_MODEL.predict_proba(input_df)[0][1])

        return {
            "prob":            round(display_prob * 100, 1),
            "model_prob":      round(model_prob * 100, 1),
            "challenger_prob": round(challenger_prob * 100, 1),
            "risk":            risk_label,
            "verdict":         verdict,
            "show_warning":    show_warning,
            "color":           RISK_COLOR_MAP.get(risk_label, "#6b7280"),
            "loan_amnt":       loan_amount,
            "annual_inc":      annual_inc,
            "fico":            fico,
            "int_rate":        int_rate,
            "dti":             float(form_data.get("dti", 0) or 0),
            "expected_loss":   round(expected_loss, 2),
            "expected_profit": round(expected_profit, 2),
            "override":        override,
            "override_reason": override_reason,
            "name":            form_data.get("borrower_name", "Borrower"),
            "error":           None,
        }
    except Exception as exc:
        log.exception("_score_borrower failed")
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — AUTH
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
@login_required
def index():
    if current_user.is_authenticated and current_user.role == "compliance":
        return redirect(url_for("dashboard"))
    return render_template("index.html")


@app.route("/signin", methods=["GET", "POST"])
def signin():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password) and user.is_active:
            login_user(user, remember=bool(request.form.get("remember")))
            session["user_email"] = user.email
            session["user_name"] = user.full_name
            session["user_role"] = user.role
            flash(f"Welcome back, {user.first_name}! 👋", "success")
            return redirect(url_for("index"))
        flash("Invalid email or password. Please try again.", "error")

    # Render signin page (GET, or POST with invalid credentials)
    return render_template_string("""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Sign In — AegisBank</title>
    <style>
      :root { color-scheme: dark; }
      body { margin:0; font-family:Arial,Helvetica,sans-serif; background:linear-gradient(135deg,#08101c 0%,#0f172a 55%,#111827 100%); color:#e5e7eb; min-height:100vh; display:grid; place-items:center; padding:24px }
      .card { width:min(100%,440px); background:rgba(15,23,42,0.92); border:1px solid rgba(148,163,184,0.2); border-radius:20px; padding:32px; box-shadow:0 24px 80px rgba(0,0,0,0.35) }
      h1 { margin:0 0 8px; font-size:2rem }
      p { margin:0 0 24px; color:#94a3b8 }
      .flash { padding:12px 14px; border-radius:12px; margin-bottom:14px; font-size:0.95rem }
      .flash.success { background: rgba(16,185,129,0.15); color:#a7f3d0 }
      .flash.error { background: rgba(239,68,68,0.15); color:#fecaca }
      label { display:block; margin:14px 0 8px; font-weight:600 }
      input[type="email"], input[type="password"] { width:100%; box-sizing:border-box; padding:12px 14px; border-radius:12px; border:1px solid rgba(148,163,184,0.28); background:rgba(2,6,23,0.75); color:#f8fafc }
      .row { display:flex; align-items:center; gap:10px; margin-top:14px; color:#cbd5e1 }
      .actions { margin-top:22px }
      button { border:0; border-radius:12px; padding:12px 16px; font-weight:700; cursor:pointer; background:linear-gradient(135deg,#c9a84c,#f0d78b); color:#0f172a }
      a { color:#93c5fd }
      .footer { margin-top:18px; font-size:0.95rem; color:#94a3b8 }
    </style>
  </head>
  <body>
    <main class="card">
      <h1>Sign in</h1>
      <p>Use your registered email and password to access the AegisBank risk engine.</p>
      {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
          {% for category, message in messages %}
            <div class="flash {{ category }}">{{ message }}</div>
          {% endfor %}
        {% endif %}
      {% endwith %}
      <form method="post" action="/signin">
        <label for="email">Email</label>
        <input id="email" name="email" type="email" required autocomplete="email" />

        <label for="password">Password</label>
        <input id="password" name="password" type="password" required autocomplete="current-password" />

        <div class="row">
          <input id="remember" name="remember" type="checkbox" />
          <label for="remember" style="margin:0;font-weight:500;">Remember me</label>
        </div>

        <div class="actions">
          <button type="submit">Sign In</button>
        </div>
      </form>
      <div class="footer">New here? <a href="/signup">Create an account</a></div>
    </main>
  </body>
</html>"""
        )


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        first_name       = request.form.get("first_name", "").strip()
        last_name        = request.form.get("last_name",  "").strip()
        email            = request.form.get("email", "").strip().lower()
        role             = request.form.get("role", "analyst")
        password         = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        agree_terms      = request.form.get("agree_terms")

        if role not in ("analyst", "risk_manager", "compliance"):
            role = "analyst"

        if not first_name or not last_name:
            flash("Please provide your full name.", "error")
        elif not email or "@" not in email:
            flash("Please provide a valid email address.", "error")
        elif User.query.filter_by(email=email).first():
            flash("An account with this email already exists. Please sign in.", "error")
        elif len(password) < 8:
            flash("Password must be at least 8 characters.", "error")
        elif password != confirm_password:
            flash("Passwords do not match. Please try again.", "error")
        elif not agree_terms:
            flash("You must agree to the Terms of Service to continue.", "error")
        else:
            new_user = User(first_name=first_name, last_name=last_name,
                            email=email, role=role, email_verified=False)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()

            token      = _generate_token(email, salt="email-verify")
            verify_url = url_for("verify_email", token=token, _external=True)
            _send_email(
                to=email,
                subject="Verify your AegisBank account",
                html_body=(
                    f"<p>Hi {first_name},</p>"
                    f"<p>Click the link below to verify your email (expires in 1 hour).</p>"
                    f"<p><a href='{verify_url}'>Verify Email</a></p>"
                    f"<p>Or copy: <code>{verify_url}</code></p>"
                    f"<p>— AegisBank AI Risk Engine</p>"
                ),
            )
            login_user(new_user)
            session["user_email"] = new_user.email
            session["user_name"]  = new_user.full_name
            session["user_role"]  = new_user.role
            flash(f"Account created! Verification email sent to {email}. 🎉", "success")
            return redirect(url_for("index"))

    return render_template("signup.html")


@app.route("/signout")
def signout():
    logout_user()
    session.clear()
    flash("You have been signed out successfully.", "success")
    return redirect(url_for("signin"))


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        user  = User.query.filter_by(email=email).first()
        flash("If an account with that email exists, a reset link has been sent.", "success")
        if user:
            token     = _generate_token(email, salt="password-reset")
            reset_url = url_for("reset_password", token=token, _external=True)
            _send_email(
                to=email,
                subject="AegisBank — Reset your password",
                html_body=(
                    f"<p>Hi {user.first_name},</p>"
                    f"<p><a href='{reset_url}'>Reset Password</a> (expires in 1 hour)</p>"
                    f"<p>If you did not request this, ignore this email.</p>"
                ),
            )
        return redirect(url_for("signin"))
    return render_template("forgot_password.html")


@app.route("/verify-email/<token>")
def verify_email(token: str):
    email = _verify_token(token, salt="email-verify", max_age=3600)
    if not email:
        flash("The verification link is invalid or has expired.", "error")
        return redirect(url_for("signin"))
    user = User.query.filter_by(email=email).first()
    if not user:
        flash("Account not found.", "error")
        return redirect(url_for("signin"))
    if not user.email_verified:
        user.email_verified = True
        db.session.commit()
        flash("Email verified! ✅", "success")
    else:
        flash("Email already verified.", "success")
    return redirect(url_for("signin"))


@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token: str):
    email = _verify_token(token, salt="password-reset", max_age=3600)
    if not email:
        flash("Reset link is invalid or expired.", "error")
        return redirect(url_for("forgot_password"))
    user = User.query.filter_by(email=email).first()
    if not user:
        flash("Account not found.", "error")
        return redirect(url_for("signin"))
    if request.method == "POST":
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")
        if len(password) < 8:
            flash("Password must be at least 8 characters.", "error")
        elif password != confirm:
            flash("Passwords do not match.", "error")
        else:
            user.set_password(password)
            db.session.commit()
            flash("Password reset successfully! ✅", "success")
            return redirect(url_for("signin"))
    return render_template("reset_password.html", token=token, email=email)


@app.route("/resend-verification")
@login_required
def resend_verification():
    if current_user.email_verified:
        flash("Your email is already verified.", "success")
        return redirect(url_for("index"))
    token      = _generate_token(current_user.email, salt="email-verify")
    verify_url = url_for("verify_email", token=token, _external=True)
    _send_email(
        to=current_user.email,
        subject="Verify your AegisBank account",
        html_body=f"<p><a href='{verify_url}'>Verify Email</a></p>",
    )
    flash("Verification email resent!", "success")
    return redirect(url_for("index"))


@app.route("/auth/google")
def auth_google():
    flash("Google Sign-In is not yet configured. Use email & password.", "error")
    return redirect(url_for("signin"))


@app.route("/auth/microsoft")
def auth_microsoft():
    flash("Microsoft Sign-In is not yet configured. Use email & password.", "error")
    return redirect(url_for("signin"))


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/batch")
@role_required("analyst", "risk_manager", "admin")
def batch():
    return render_template("batch.html")


@app.route("/batch/template")
@login_required
def batch_template():
    csv_content = (
        "loan_amnt,funded_amnt,int_rate,installment,annual_inc,dti,"
        "fico_range_low,fico_range_high,open_acc,pub_rec,revol_bal,"
        "revol_util,total_acc,loan_status\n"
        "10000,10000,12.5,335.54,65000,15.2,680,684,8,0,5000,30.5,22,\n"
        "25000,25000,18.9,650.20,45000,28.7,620,624,12,1,12000,68.2,18,\n"
    )
    return app.response_class(
        csv_content,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=aegisbank_template.csv"},
    )


@app.route("/compare", methods=["GET", "POST"])
@role_required("analyst", "risk_manager", "admin")
def compare():
    if MODEL is None:
        flash("Model not loaded — run train_model.py first.", "error")
        return redirect(url_for("dashboard"))

    result_a = result_b = None
    form_a   = form_b   = {}

    if request.method == "POST":
        raw    = request.form.to_dict()
        form_a = {k[2:]: v for k, v in raw.items() if k.startswith("a_")}
        form_b = {k[2:]: v for k, v in raw.items() if k.startswith("b_")}
        result_a = _score_borrower(form_a)
        result_b = _score_borrower(form_b)

    return render_template("compare.html",
                           result_a=result_a, result_b=result_b,
                           form_a=form_a, form_b=form_b)


# ─────────────────────────────────────────────────────────────────────────────
# SOCKETIO — MAIN PREDICTION ROUTE
# ─────────────────────────────────────────────────────────────────────────────

@socketio.on("submit_prediction")
def handle_prediction(form_data):
    if MODEL is None:
        emit("prediction_error", {"error": "Model not loaded — run train_model.py first."})
        return

    emit("progress", {"step": "Validating inputs...", "percent": 10})
    time.sleep(0.3)

    errors = _validate_input(form_data)
    if errors:
        emit("prediction_error", {"error": "\n".join(errors)})
        return

    emit("progress", {"step": "Running XGBoost model...", "percent": 40})
    time.sleep(0.3)

    try:
        # ── Step 1: preprocess & feature engineering ─────────────────────────
        input_df = preprocess_input(form_data)
        input_df = create_features_live(input_df)
        # SHAP must be called AFTER reindex so columns match the model
        input_df = input_df.reindex(columns=MODEL_FEATURES, fill_value=0.0)

        emit("progress", {"step": "Computing SHAP values...", "percent": 60})
        time.sleep(0.3)

        # ── Step 2: SHAP explanation ─────────────────────────────────────────
        # explain_single returns real per-prediction SHAP values, not hardcoded values
        explanation = EXPLAINER.explain_single(input_df)

        # ── Step 3: fairness checks ──────────────────────────────────────────
        fairness_flag    = EXPLAINER.check_individual_fairness(form_data)
        bias_flag        = EXPLAINER.check_group_bias(form_data)
        sensitive_warning = EXPLAINER.validate_sensitive_features(form_data)

        emit("progress", {"step": "Applying underwriting rules...", "percent": 75})
        time.sleep(0.3)

        # ── Step 4: model inference ──────────────────────────────────────────
        model_prob = float(MODEL.predict_proba(input_df)[0][1])

        # Shadow model (A/B testing)
        challenger_prob = 0.0
        if CHALLENGER_MODEL:
            challenger_prob = float(CHALLENGER_MODEL.predict_proba(input_df)[0][1])

        log.info("Probability: %.4f | Challenger: %.4f", model_prob, challenger_prob)

        # ── Step 5: override rules ───────────────────────────────────────────
        override, override_reason, adj_prob = check_overrides(form_data)

        # When override fires, display probability is rule-based (not misleading model score)
        display_prob = max(adj_prob or model_prob, model_prob) if override else model_prob

        # ── Step 6: financial calculations ───────────────────────────────────
        loan_amount = float(form_data.get("loan_amnt",      0) or 0)
        annual_inc  = float(form_data.get("annual_inc",     0) or 0)
        fico        = float(form_data.get("fico_range_low", 0) or 0)
        int_rate    = float(form_data.get("int_rate",       0) or 0)
        dti         = float(form_data.get("dti",            0) or 0)

        lgd             = calculate_lgd(fico)
        ead             = loan_amount
        expected_loss   = display_prob * lgd * ead
        expected_profit = calculate_expected_profit(loan_amount, int_rate, display_prob, lgd)

        # ── Step 7: risk classification ──────────────────────────────────────
        risk_label, verdict, show_warning = classify_risk(
            display_prob, override, fico, loan_amount, annual_inc
        )

        if override:
            message = f"⛔ Hard Decline — {override_reason}"
        elif show_warning:
            message = "Default Risk Detected — Enhanced Review Required"
        else:
            message = "Safe Borrower — No Immediate Risk"

        risk_note = (
            "📌 Credit Invisible — evaluated using alternative data"
            if fico == 0 else "Standard credit evaluation"
        )

        # Load Youden's J threshold (not hardcoded 0.4 or 0.5)
        threshold = _load_threshold()

        emit("progress", {"step": "Building audit record...", "percent": 90})
        time.sleep(0.2)

        # ── Step 8: build and persist record ─────────────────────────────────
        record = {
            "id":               str(uuid.uuid4()),
            "trace_id":         str(uuid.uuid4()),
            "timestamp":        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "user_email":       current_user.email if hasattr(current_user, "email") else "api",
            "user_role":        current_user.role  if hasattr(current_user, "role")  else "api",
            "borrower":         form_data.get("borrower_name", "Anonymous"),
            "addr_state":       form_data.get("addr_state", ""),
            "loan_amnt":        loan_amount,
            "int_rate":         int_rate,
            "annual_inc":       annual_inc,
            "dti":              dti,
            "fico":             fico,
            "purpose":          form_data.get("purpose", ""),
            "grade":            form_data.get("grade", ""),
            "prediction":       verdict,
            "prediction_numeric": 1 if verdict == "Default" else 0,
            "verdict":          verdict,
            "decision":         verdict,
            "probability":      round(display_prob * 100, 2),
            "model_probability": round(model_prob * 100, 2),
            "PD":               round(display_prob, 4),
            "LGD":              round(lgd, 2),
            "EAD":              round(ead, 2),
            "expected_loss":    round(expected_loss, 2),
            "expected_profit":  round(expected_profit, 2),
            "model_version":    "v1.0",
            "decision_threshold": threshold,
            "top_features":     explanation,      # real SHAP values per prediction
            "fairness_check":   fairness_flag,
            "bias_check":       bias_flag,
            "sensitive_warning": sensitive_warning,
            "challenger_prob":  round(challenger_prob * 100, 2),
            "risk_level":       risk_label,
            "show_warning":     show_warning,
            "message":          message,
            "override_triggered": override,
            "override_reason":  override_reason,
            "color":            RISK_COLOR_MAP.get(risk_label, "#6b7280"),
            "risk_note":        risk_note,
            "raw_input":        form_data,
            "actual_outcome":   None,
        }

        report      = generate_risk_report(record)
        report_path = save_report(report, record["id"])
        record["report_path"] = report_path

        _append_to_history(record)
        log_decision(record)

        # ── Step 9: background MLOps ─────────────────────────────────────────
        # Run in a daemon thread so the socket response is not delayed
        def _background_mlops():
            try:
                # Scheduled retraining every 100 predictions
                if should_retrain():
                    log.info("⚡ Scheduled retraining (every 100 predictions)…")
                    feedback_data = build_feedback_dataset()
                    if feedback_data is not None:
                        update_training_data(feedback_data)
                        log.info("🔁 Feedback data appended to training set")
                    retrain_model()
                    reload_model()

                # Drift detection
                from monitoring.drift_detection import detect_drift
                current_data = get_current_data()
                if current_data is not None:
                    _, drift_flag = detect_drift(REFERENCE_DATA, current_data)
                    if drift_flag:
                        log.warning("🚨 DRIFT DETECTED — triggering retraining")
                        retrain_model()
                        reload_model()

            except Exception as exc:
                log.error("Background MLOps error: %s", exc)

        import threading
        threading.Thread(target=_background_mlops, daemon=True).start()

        emit("progress", {"step": "Decision ready ✓", "percent": 100})
        time.sleep(0.1)
        emit("prediction_complete", {"record_id": record["id"]})

    except Exception as exc:
        log.exception("Prediction error")
        emit("prediction_error", {"error": f"Prediction failed: {exc}"})


@app.route("/result/<record_id>")
@fl_login_required
def prediction_result(record_id: str):
    records = _load_history()
    record  = next((r for r in records if r.get("id") == record_id), None)
    if not record:
        abort(404)
    return render_template(
        "result.html",
        risk         = record["risk_level"],
        show_warning = record["show_warning"],
        prob         = record["probability"] / 100.0,
        model_prob   = record.get("model_probability", record["probability"]) / 100.0,
        override_triggered = record.get("override_triggered", False),
        override_reason    = record.get("override_reason"),
        verdict      = record["verdict"],
        record       = record,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — DASHBOARD, HISTORY, REPORTS, ADMIN
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/dashboard")
@login_required
def dashboard():
    has_api_key = False
    if current_user.is_authenticated:
        has_api_key = ApiKey.query.filter_by(user_id=current_user.id).first() is not None
    return render_template("dashboard.html", metrics=METRICS, has_api_key=has_api_key)


@app.route("/history")
@login_required
def history():
    return render_template("history.html", records=_load_history())


@app.route("/reports")
@login_required
def reports():
    return render_template("reports.html", records=_load_history())


@app.route("/reports/<record_id>")
@login_required
def report_detail(record_id: str):
    record = next((r for r in _load_history() if r.get("id") == record_id), None)
    if record is None:
        abort(404)
    return render_template("report_detail.html", record=record)


@app.route("/admin")
@role_required("admin")
def admin_panel():
    users = User.query.order_by(User.created_at).all()
    return render_template("admin.html", users=users,
                           history_count=len(_load_history()),
                           model_features=len(MODEL_FEATURES),
                           model_loaded=MODEL is not None)


@app.route("/admin/promote", methods=["POST"])
@role_required("admin")
def admin_promote():
    email    = request.form.get("email", "").strip().lower()
    new_role = request.form.get("role", "analyst")
    user = User.query.filter_by(email=email).first()
    if user:
        user.role = new_role
        db.session.commit()
        if email == current_user.email:
            session["user_role"] = new_role
        flash(f"Role updated: {email} → {new_role}", "success")
    else:
        flash(f"User not found: {email}", "error")
    return redirect(url_for("admin_panel"))


@app.route("/audit")
@role_required("compliance", "admin")
def audit():
    records = sorted(_load_history(), key=lambda x: x.get("timestamp", ""), reverse=True)
    return render_template("audit.html", records=records)


@app.route("/heatmap")
@login_required
def heatmap():
    return render_template("heatmap.html")


@app.route("/timeline")
@login_required
def timeline():
    query = request.args.get("q", "").strip()
    return render_template("timeline.html", query=query)


# ─────────────────────────────────────────────────────────────────────────────
# JSON APIs
# ─────────────────────────────────────────────────────────────────────────────

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
            if q in r.get("borrower",   "").lower()
            or q in r.get("purpose",    "").lower()
            or q in r.get("risk_level", "").lower()
        ]
    return jsonify(records)


@app.route("/api/timeline/<path:borrower_name>")
@login_required
def api_timeline(borrower_name: str):
    records    = _load_history()
    name_lower = borrower_name.lower()
    matches    = sorted(
        [r for r in records if r.get("borrower", "").lower() == name_lower],
        key=lambda r: r.get("timestamp", ""),
    )
    if not matches:
        return jsonify({"error": f"No records found for '{borrower_name}'"}), 404

    points = [
        {
            "index":       i + 1,
            "timestamp":   r.get("timestamp", ""),
            "probability": round(float(r.get("probability", 0)), 1),
            "risk_level":  r.get("risk_level", ""),
            "verdict":     r.get("verdict", ""),
            "loan_amnt":   r.get("loan_amnt", 0),
            "int_rate":    r.get("int_rate", 0),
            "fico":        r.get("fico", 0),
            "annual_inc":  r.get("annual_inc", 0),
            "purpose":     r.get("purpose", ""),
            "id":          r.get("id", ""),
        }
        for i, r in enumerate(matches)
    ]

    delta = points[-1]["probability"] - points[0]["probability"] if len(points) >= 2 else 0
    trend = "worsening" if delta > 5 else ("improving" if delta < -5 else "stable")
    if len(points) < 2:
        trend = "single"

    return jsonify({
        "borrower": borrower_name,
        "count":    len(points),
        "trend":    trend,
        "delta":    round(delta, 1),
        "latest":   points[-1],
        "first":    points[0],
        "points":   points,
    })


@app.route("/api/borrower-names")
@login_required
def api_borrower_names():
    q       = request.args.get("q", "").lower()
    records = _load_history()
    names   = sorted({
        r.get("borrower", "")
        for r in records
        if r.get("borrower") and q in r.get("borrower", "").lower()
    })
    return jsonify(names[:20])


@app.route("/api/geo-risk")
@login_required
def api_geo_risk():
    _BASELINE = {
        "AL": 52, "AK": 41, "AZ": 48, "AR": 55, "CA": 44, "CO": 38, "CT": 42,
        "DE": 40, "FL": 51, "GA": 50, "HI": 36, "ID": 39, "IL": 46, "IN": 49,
        "IA": 37, "KS": 43, "KY": 54, "LA": 58, "ME": 38, "MD": 43, "MA": 39,
        "MI": 47, "MN": 36, "MS": 61, "MO": 48, "MT": 41, "NE": 38, "NV": 53,
        "NH": 35, "NJ": 44, "NM": 52, "NY": 45, "NC": 49, "ND": 34, "OH": 48,
        "OK": 53, "OR": 40, "PA": 44, "RI": 43, "SC": 51, "SD": 37, "TN": 52,
        "TX": 49, "UT": 37, "VT": 34, "VA": 42, "WA": 39, "WV": 57, "WI": 40,
        "WY": 41, "DC": 47, "PR": 63, "VI": 59,
    }
    records    = _load_history()
    state_data: dict = {}
    for r in records:
        state = r.get("addr_state") or (r.get("raw_input") or {}).get("addr_state", "")
        state = str(state).strip().upper()
        if len(state) != 2:
            continue
        prob = float(r.get("probability", 0))
        if state not in state_data:
            state_data[state] = {"sum": 0.0, "count": 0}
        state_data[state]["sum"]   += prob
        state_data[state]["count"] += 1

    result = []
    for state, baseline in _BASELINE.items():
        if state in state_data and state_data[state]["count"] > 0:
            avg = round(state_data[state]["sum"] / state_data[state]["count"], 1)
            cnt = state_data[state]["count"]
            src = "live"
        else:
            avg = round(baseline + (hash(state) % 7) - 3, 1)
            cnt = 0
            src = "baseline"
        result.append({"state": state, "avg_prob": avg, "count": cnt, "source": src})

    return jsonify(result)


@app.route("/api/history/confirm", methods=["POST"])
@login_required
def confirm_outcome():
    data      = request.json
    record_id = data.get("id")
    outcome   = data.get("outcome")
    if record_id is None or outcome is None:
        return jsonify({"error": "Missing ID or outcome"}), 400
    try:
        history = _load_history()
        updated = False
        for entry in history:
            if entry.get("id") == record_id:
                entry["actual_outcome"] = outcome
                updated = True
                break
        if updated:
            _save_history(history)
            try:
                from monitoring.model_health import monitor_health
                monitor_health()
            except ImportError:
                pass
            return jsonify({"status": "success"})
        return jsonify({"error": "Record not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API KEY & REST PREDICT ────────────────────────────────────────────────────

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            auth = request.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                api_key = auth.split(" ")[1]
        if not api_key:
            return jsonify({"error": "Missing API Key."}), 401
        valid_user = next(
            (k.user for k in ApiKey.query.all() if k.check_key(api_key)), None
        )
        if not valid_user:
            return jsonify({"error": "Invalid API Key."}), 403
        return f(*args, **kwargs)
    return decorated


@app.route("/api/v1/keys/generate", methods=["POST"])
@fl_login_required
def generate_api_key():
    raw_key = secrets.token_urlsafe(32)
    ApiKey.query.filter_by(user_id=current_user.id).delete()
    new_key = ApiKey(user_id=current_user.id)
    new_key.set_key(raw_key)
    db.session.add(new_key)
    db.session.commit()
    return jsonify({"message": "API key generated.", "api_key": raw_key})


@app.route("/api/v1/predict", methods=["POST"])
@require_api_key
def api_predict():
    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 503
    form_data = request.json
    if not form_data:
        return jsonify({"error": "Invalid JSON payload"}), 400
    errors = _validate_input(form_data)
    if errors:
        return jsonify({"error": "Validation failed", "details": errors}), 400
    try:
        result = _score_borrower(form_data)
        if result.get("error"):
            return jsonify({"error": result["error"]}), 500
        return jsonify({
            "status": "success",
            "prediction": {
                "risk_level":       result["risk"],
                "probability":      result["prob"],
                "model_probability": result["model_prob"],
                "verdict":          result["verdict"],
                "expected_loss":    result["expected_loss"],
                "expected_profit":  result["expected_profit"],
                "override_triggered": result["override"],
                "override_reason":  result["override_reason"],
            },
        })
    except Exception as exc:
        log.exception("API Prediction error")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/v1/mlops/health")
@role_required("admin", "risk_manager")
def api_mlops_health():
    try:
        from monitoring.drift_detection import run_monitoring
        drift_results     = run_monitoring()
        champion_metrics  = json.load(open(METRICS_PATH)) if os.path.exists(METRICS_PATH) else {}
        challenger_metrics = (
            json.load(open(CHALLENGER_METRICS_PATH))
            if os.path.exists(CHALLENGER_METRICS_PATH) else {}
        )
        live_accuracy = None
        if os.path.exists(HISTORY_PATH):
            df = pd.DataFrame(_load_history())
            if "actual_outcome" in df.columns:
                valid = df.dropna(subset=["actual_outcome"])
                if len(valid) > 0:
                    live_accuracy = round(
                        float((valid["actual_outcome"] == valid["verdict"]).mean() * 100), 2
                    )
        return jsonify({
            "drift": drift_results,
            "champion": champion_metrics,
            "challenger": challenger_metrics,
            "live_accuracy": live_accuracy,
            "retrain_threshold": 80.0,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/mlops/retrain", methods=["POST"])
@role_required("admin")
def api_mlops_retrain():
    try:
        retrain_model()
        reload_model()
        return jsonify({"status": "success", "message": "Retraining complete."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/mlops/reload", methods=["POST"])
@role_required("admin")
def api_mlops_reload():
    try:
        reload_model()
        return jsonify({"status": "success", "message": "Models reloaded."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
    with app.app_context():
        db.create_all()
        _seed_default_users()
        log.info("✅ Database ready at %s", DB_PATH)

    socketio.run(app, debug=False, host="127.0.0.1", port=5000,
                 allow_unsafe_werkzeug=True)
