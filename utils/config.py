# utils/config.py
"""
Central configuration for AegisBank Loan Default Prediction System.
All paths, hyperparameters, and constants live here.
"""

import os

# ── BASE DIRECTORY ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── DATA PATHS ───────────────────────────────────────────────────────────────
RAW_DATA_PATH       = os.path.join(BASE_DIR, "data", "raw",       "loan_dataset.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")

# ── MODEL ARTIFACTS ──────────────────────────────────────────────────────────
CHAMPION_MODEL_PATH    = os.path.join(BASE_DIR, "models", "champion_model.pkl")
CHALLENGER_MODEL_PATH  = os.path.join(BASE_DIR, "models", "challenger_model.pkl")
MODEL_PATH             = CHAMPION_MODEL_PATH  # Compatibility
FEATURES_PATH          = os.path.join(BASE_DIR, "utils",  "model_features.pkl")
METRICS_PATH           = os.path.join(BASE_DIR, "model_metrics.json")
CHALLENGER_METRICS_PATH = os.path.join(BASE_DIR, "challenger_metrics.json")

# ── PREDICTION HISTORY ───────────────────────────────────────────────────────
HISTORY_PATH       = os.path.join(BASE_DIR, "outputs", "prediction_history.json")

# ── TARGET & SENSITIVE COLUMNS ───────────────────────────────────────────────
TARGET_COLUMN      = "loan_status"
SENSITIVE_COLUMN   = "addr_state"

# ── TRAIN / TEST SPLIT ───────────────────────────────────────────────────────
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# ── XGBOOST HYPERPARAMETERS ─────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":    100,
    "max_depth":         5,
    "learning_rate":   0.1,
    "subsample":       0.8,
    "colsample_bytree":0.8,
    "eval_metric":     "logloss",
    "random_state":    RANDOM_STATE,
    "use_label_encoder": False,
}

# ── RISK THRESHOLDS ──────────────────────────────────────────────────────────
RISK_LEVELS = [
    (0.30, "LOW RISK",       "Approve", "#22c55e"),
    (0.50, "MEDIUM RISK",    "Review",  "#f59e0b"),
    (0.70, "HIGH RISK",      "Decline", "#f97316"),
    (1.01, "VERY HIGH RISK", "Decline", "#ef4444"),
]

def get_risk_level(probability: float) -> dict:
    """Return risk label, verdict, and color for a given probability (0–1)."""
    for threshold, label, verdict, color in RISK_LEVELS:
        if probability < threshold:
            return {"label": label, "verdict": verdict, "color": color}
    return {"label": "VERY HIGH RISK", "verdict": "Decline", "color": "#ef4444"}