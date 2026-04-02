# src/train_model.py
"""
Loan Default Prediction — Model Training

Steps:
  1. Load processed CSV
  2. One-hot encode & sanitize column names (XGBoost-safe)
  3. Train-test split
  4. Train Logistic Regression, Random Forest, XGBoost
  5. Evaluate & pick best model by ROC-AUC
  6. Save model + feature list + metrics JSON
"""

import sys
import os
import re
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import (
    PROCESSED_DATA_PATH, TARGET_COLUMN,
    TEST_SIZE, RANDOM_STATE,
    MODEL_PATH, FEATURES_PATH, METRICS_PATH, XGB_PARAMS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def sanitize_columns(columns) -> list:
    """
    Make column names safe for XGBoost:
      - Replace forbidden chars [ ] < >
      - Replace whitespace with _
      - Keep only alphanumeric + _
      - Deduplicate by appending _N
    """
    seen: dict = {}
    result: list = []
    for col in columns:
        c = re.sub(r"[\[\]<>]", "_", str(col))
        c = re.sub(r"\s+",      "_", c.strip())
        c = re.sub(r"[^0-9a-zA-Z_]", "_", c)
        if c in seen:
            seen[c] += 1
            c = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
        result.append(c)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    log.info("Loaded data: %s rows × %s cols", *df.shape)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # One-hot encode remaining categoricals
    X = pd.get_dummies(X, drop_first=True)
    X.columns = sanitize_columns(X.columns)
    X = X.astype("float32")

    log.info("After encoding: %s features", X.shape[1])
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 2. SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def train_all(X_train, y_train) -> dict:
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=5000, solver="saga", random_state=RANDOM_STATE),
        "random_forest":       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "xgboost":             XGBClassifier(**XGB_PARAMS),
    }
    trained = {}
    for name, model in candidates.items():
        log.info("Training %s …", name)
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


# ─────────────────────────────────────────────────────────────────────────────
# 4. EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(models: dict, X_test, y_test) -> tuple[dict, dict]:
    """Return (metrics_per_model, scores_for_comparison)."""
    all_metrics: dict = {}
    scores: dict = {}

    for name, model in models.items():
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        metrics = {
            "accuracy":  round(float(accuracy_score(y_test, preds)),            4),
            "precision": round(float(precision_score(y_test, preds,  zero_division=0)), 4),
            "recall":    round(float(recall_score(y_test, preds,     zero_division=0)), 4),
            "f1_score":  round(float(f1_score(y_test, preds,         zero_division=0)), 4),
            "roc_auc":   round(float(roc_auc_score(y_test, probs)),              4),
            "confusion_matrix": {
                "tn": int(tn), "fp": int(fp),
                "fn": int(fn), "tp": int(tp),
            },
        }

        log.info("%-22s  acc=%.4f  roc=%.4f", name, metrics["accuracy"], metrics["roc_auc"])
        log.info("\n%s", classification_report(y_test, preds))

        all_metrics[name] = metrics
        scores[name] = metrics["roc_auc"]

    return all_metrics, scores


# ─────────────────────────────────────────────────────────────────────────────
# 5. SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_artifacts(models: dict, all_metrics: dict, scores: dict, feature_names: list) -> None:
    best_name = max(scores, key=scores.get)
    best_model = models[best_name]
    log.info("Best model: %s  (roc_auc=%.4f)", best_name, scores[best_name])

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    log.info("Model saved → %s", MODEL_PATH)

    import pickle
    os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(feature_names, f)
    log.info("Feature list saved → %s", FEATURES_PATH)

    best_metrics = all_metrics[best_name]
    best_metrics["model_name"] = best_name
    with open(METRICS_PATH, "w") as f:
        json.dump(best_metrics, f, indent=4)
    log.info("Metrics saved → %s", METRICS_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    X, y         = load_and_preprocess()
    X_train, X_test, y_train, y_test = split(X, y)

    models           = train_all(X_train, y_train)
    all_metrics, scores = evaluate_all(models, X_test, y_test)

    save_artifacts(models, all_metrics, scores, list(X.columns))
    log.info("Training pipeline complete ✅")


if __name__ == "__main__":
    main()