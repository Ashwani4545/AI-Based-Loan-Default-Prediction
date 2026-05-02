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
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score, roc_curve,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import (
    PROCESSED_DATA_PATH, TARGET_COLUMN,
    TEST_SIZE, RANDOM_STATE,
    CHAMPION_MODEL_PATH, CHALLENGER_MODEL_PATH, FEATURES_PATH, 
    METRICS_PATH, CHALLENGER_METRICS_PATH, XGB_PARAMS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION: Alternative Data Sources
# ─────────────────────────────────────────────────────────────────────────────
USE_REAL_ALTERNATIVE_DATA = True  # Toggle to True for production with real data
ALTERNATIVE_DATA_PATH = os.path.join(Path(__file__).resolve().parent.parent, "data", "alternative_data.csv")

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


def calculate_profit(y_true, y_pred, loan_amounts):
    profit = 0

    for yt, yp, loan in zip(y_true, y_pred, loan_amounts):
        if yp == 0:  # predicted repay
            if yt == 0:
                profit += loan * 0.1   # interest gain
            else:
                profit -= loan         # default loss
        else:
            profit += 0  # rejected loan

    return profit


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    # Financial ratios
    df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1e-6)
    df["installment_to_income"] = df["installment"] / (df["annual_inc"] + 1e-6)

    # Credit behavior
    df["credit_utilization"] = df["revol_bal"] / (df["revol_bal"] + df["bc_open_to_buy"] + 1e-6)

    # Behavioral features
    df["payment_capacity"] = df["annual_inc"] - df["installment"] * 12
    df["credit_stress"] = df["dti"] * df["loan_amnt"]
    df["recent_inquiries_flag"] = (df["inq_last_6mths"] > 3).astype(int)

    # Risk indicators
    df["high_dti_flag"] = (df["dti"] > 20).astype(int)
    df["low_fico_flag"] = (df["fico_range_low"] < 600).astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────

def _load_alternative_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load alternative credit data for credit-invisible users.
    Supports both real and synthetic data sources.
    """
    use_real_alternative_data = USE_REAL_ALTERNATIVE_DATA

    if use_real_alternative_data:
        try:
            alt_df = pd.read_csv(ALTERNATIVE_DATA_PATH)
            log.info("Loaded real alternative data: %s rows", len(alt_df))
            # Merge on common ID (adjust key as needed)
            if "customer_id" in alt_df.columns and "customer_id" in df.columns:
                df = df.merge(alt_df, on="customer_id", how="left")
            elif "id" in alt_df.columns and "id" in df.columns:
                df = df.merge(alt_df, on="id", how="left")
            else:
                log.warning("Cannot merge alternative data — no common ID column. Using synthetic fallback.")
                use_real_alternative_data = False
        except FileNotFoundError:
            log.warning("Alternative data file not found at %s. Falling back to synthetic data.", ALTERNATIVE_DATA_PATH)
            use_real_alternative_data = False
    
    # Use synthetic as fallback or primary
    if not use_real_alternative_data:
<<<<<<< HEAD
        log.info("Using placeholder (0) for alternative features — no real alternative data available.")
        log.info("FIX Bug 8: previously used np.random noise here which trained the model on garbage.")
        log.info("Now using 0 consistently — matches what inference sends when these fields are absent.")
        # NOTE: To properly use these features, collect them in the web form
        # (mobile_usage_score, digital_txn_count, utility_payment_score, employment_stability)
        # and provide real data. Until then, 0 is the honest placeholder.
        df["mobile_usage_score"]    = 0
        df["digital_txn_count"]     = 0
        df["utility_payment_score"] = 0
        df["employment_stability"]  = 0
=======
        # Bug #8 fix: random noise features removed — they pollute the model.
        # If real alternative data is needed, provide it via alternative_data.csv.
        log.info("No real alternative data available — skipping alternative features")
>>>>>>> 44ab82bb832d0cf735042468c185eb3463bf6a67
    
    return df


def load_and_preprocess():
    df = pd.read_csv(PROCESSED_DATA_PATH)
<<<<<<< HEAD
    log.info("Loaded data: %s rows × %s cols", *df.shape)

    # Economic context features (static demo values)
    df["inflation_rate"] = 0.06
    df["interest_rate_env"] = 0.08
    df["unemployment_rate"] = 0.07
    df["economic_stress"] = (
        df["inflation_rate"] * 0.4 +
        df["unemployment_rate"] * 0.4 +
        df["interest_rate_env"] * 0.2
    )
=======
    log.info("Loaded data: %d rows — %d cols", len(df), len(df.columns))
    
    # Subsample for speed in demo environment
    df = df.sample(n=min(10000, len(df)), random_state=RANDOM_STATE)
    log.info("Subsampled to %d rows for faster training", len(df))

    # Bug #9 fix: Economic context features removed.
    # Hardcoded constants (0.06, 0.08, 0.07) are identical for every row,
    # so the model learns zero information from them.
>>>>>>> 44ab82bb832d0cf735042468c185eb3463bf6a67

    # Load alternative credit data (real or synthetic)
    df = _load_alternative_data(df)

    # Create engineered features before training
    df = create_features(df)

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
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        SMOTE = None

    if SMOTE is not None:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        log.info("SMOTE applied: %d -> %d samples", len(X_train), len(X_train_res))
    else:
        X_train_res, y_train_res = X_train, y_train
        log.warning("imblearn not installed; training without SMOTE.")

    counter = Counter(y_train_res)
    scale_pos_weight = counter[0] / counter[1]

    xgb_base = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",  # BEST for imbalance
    )
    xgb_param_grid = {
        "n_estimators": [150, 200],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
    }
    xgb_grid = GridSearchCV(
        estimator=xgb_base,
        param_grid=xgb_param_grid,
        scoring="recall",
        cv=3,
        n_jobs=-1,
        verbose=0,
    )
    xgb_grid.fit(X_train_res, y_train_res)
    log.info("Best XGBoost params (recall): %s", xgb_grid.best_params_)

    candidates = {
<<<<<<< HEAD
        "logistic_regression": LogisticRegression(max_iter=5000, solver="saga", random_state=RANDOM_STATE),
        "random_forest":       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
=======
        "logistic_regression": LogisticRegression(max_iter=100, solver="lbfgs", random_state=RANDOM_STATE),
        "random_forest":       RandomForestClassifier(n_estimators=20, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1),
>>>>>>> 44ab82bb832d0cf735042468c185eb3463bf6a67
        "xgboost":             xgb_grid.best_estimator_,
    }
    trained = {}
    for name, model in candidates.items():
        log.info("Training %s …", name)
        model.fit(X_train_res, y_train_res)
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
        y_prob = probs

        # ROC curve threshold tuning
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        best_threshold = thresholds[(tpr - fpr).argmax()]
        print(f"Best Threshold: {best_threshold:.6f}")

        # Classification metrics
        recall = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)

        print(f"\n{name} - Classification Metrics")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")

        # Regression-style probability error metrics
        mse = mean_squared_error(y_test, y_prob)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_prob)
        mape = np.mean(np.abs((y_test - y_prob) / (y_test + 1e-10))) * 100
        r2 = r2_score(y_test, y_prob)

        print(f"{name} - Regression Metrics")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"MAPE: {mape:.4f}")
        print(f"R2: {r2:.6f}")

        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        loan_amounts = X_test["loan_amnt"]
        profit = calculate_profit(y_test, preds, loan_amounts)
        metrics = {
            "accuracy":  round(float(accuracy_score(y_test, preds)),            4),
            "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
            "recall":    round(float(recall), 4),
            "f1_score":  round(float(f1), 4),
            "roc_auc":   round(float(roc_auc), 4),
            "mse":       round(float(mse), 6),
            "rmse":      round(float(rmse), 6),
            "mae":       round(float(mae), 6),
            "mape":      round(float(mape), 4),
            "r2":        round(float(r2), 6),
            "profit":    round(float(profit), 2),
            "confusion_matrix": {
                "tn": int(tn), "fp": int(fp),
                "fn": int(fn), "tp": int(tp),
            },
        }

        log.info("%-22s  recall=%.4f  f1=%.4f", name, metrics["recall"], metrics["f1_score"])
        log.info("\n%s", classification_report(y_test, preds))

        all_metrics[name] = metrics
        scores[name] = metrics["roc_auc"]

    return all_metrics, scores


# ─────────────────────────────────────────────────────────────────────────────
# 5. SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_artifacts(models: dict, all_metrics: dict, scores: dict, feature_names: list) -> None:
    # Sort models by profit (our primary selection metric)
    sorted_model_names = sorted(all_metrics, key=lambda m: all_metrics[m]["profit"], reverse=True)
    
    best_name = sorted_model_names[0]
    challenger_name = sorted_model_names[1] if len(sorted_model_names) > 1 else best_name
    
    log.info("🏆 Champion model: %s (profit=%.2f)", best_name, all_metrics[best_name]["profit"])
    log.info("🥈 Challenger model: %s (profit=%.2f)", challenger_name, all_metrics[challenger_name]["profit"])

    os.makedirs(os.path.dirname(CHAMPION_MODEL_PATH), exist_ok=True)
    
    # Save Champion
    joblib.dump(models[best_name], CHAMPION_MODEL_PATH)
    log.info("Champion saved → %s", CHAMPION_MODEL_PATH)

    # Save Challenger
    joblib.dump(models[challenger_name], CHALLENGER_MODEL_PATH)
    log.info("Challenger saved → %s", CHALLENGER_MODEL_PATH)

    import pickle
    os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(feature_names, f)
    log.info("Feature list saved → %s", FEATURES_PATH)

    # Save Metrics for both
    best_metrics = all_metrics[best_name]
    best_metrics["model_name"] = best_name
    with open(METRICS_PATH, "w") as f:
        json.dump(best_metrics, f, indent=4)
        
    challenger_metrics = all_metrics[challenger_name]
    challenger_metrics["model_name"] = challenger_name
    with open(CHALLENGER_METRICS_PATH, "w") as f:
        json.dump(challenger_metrics, f, indent=4)
        
    log.info("Metrics saved for both Champion and Challenger.")


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