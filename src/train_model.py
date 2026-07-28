# src/train_model.py
"""
Loan Default Prediction — Model Training (XGBoost champion)

Steps:
  1. Load processed CSV
  2. Drop high-cardinality columns that explode feature space
  3. Engineer features
  4. One-hot encode & sanitize column names (XGBoost-safe)
  5. Feature selection (top 80 by importance)
  6. Train-test split (stratified)
  7. SMOTE on training split only (no leakage)
  8. Train XGBoost with GridSearchCV
  9. Evaluate, compute Youden's J threshold
  10. Save champion model + feature list + metrics JSON
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
    roc_curve,
)
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import (
    PROCESSED_DATA_PATH, TARGET_COLUMN,
    TEST_SIZE, RANDOM_STATE,
    CHAMPION_MODEL_PATH, FEATURES_PATH,
    METRICS_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ALTERNATIVE_DATA_PATH = os.path.join(
    Path(__file__).resolve().parent.parent, "data", "alternative_data.csv"
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def sanitize_columns(columns) -> list:
    """Make column names safe for XGBoost (no [ ] < > special chars)."""
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


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered features — must exactly mirror create_features_live() in app.py."""
    df["loan_to_income"]         = df["loan_amnt"]    / (df["annual_inc"] + 1e-6)
    df["installment_to_income"]  = df["installment"]  / (df["annual_inc"] + 1e-6)
    df["credit_utilization"]     = df["revol_bal"]    / (df["revol_bal"] + df["bc_open_to_buy"] + 1e-6)
    df["payment_capacity"]       = df["annual_inc"]   - df["installment"] * 12
    df["credit_stress"]          = df["dti"]          * df["loan_amnt"]
    df["recent_inquiries_flag"]  = (df["inq_last_6mths"] > 3).astype(int)
    df["high_dti_flag"]          = (df["dti"] > 20).astype(int)
    df["low_fico_flag"]          = (df["fico_range_low"] < 600).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────

def _load_alternative_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge real alternative data if available; otherwise use 0 placeholders.
    Using 0 is honest — it matches what the inference pipeline sends when
    these fields are absent from the form, so there is no train/serve skew.
    Random noise was previously used here which polluted the model.
    """
    try:
        alt_df = pd.read_csv(ALTERNATIVE_DATA_PATH)
        log.info("Loaded real alternative data: %s rows", len(alt_df))
        if "customer_id" in alt_df.columns and "customer_id" in df.columns:
            df = df.merge(alt_df, on="customer_id", how="left")
            return df
        if "id" in alt_df.columns and "id" in df.columns:
            df = df.merge(alt_df, on="id", how="left")
            return df
        log.warning("Cannot merge alternative data — no common ID column. Using 0 placeholders.")
    except FileNotFoundError:
        log.info("No alternative_data.csv found — using 0 placeholders.")
    except Exception as exc:
        log.warning("Alternative data load failed: %s — using 0 placeholders.", exc)

    df["mobile_usage_score"]    = 0
    df["digital_txn_count"]     = 0
    df["utility_payment_score"] = 0
    df["employment_stability"]  = 0
    return df


def load_and_preprocess():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    log.info("Loaded data: %d rows × %d cols", len(df), len(df.columns))

    # Drop high-cardinality columns that explode the feature space after
    # one-hot encoding without adding predictive signal.
    # addr_state → 50 dummy cols;  sub_grade → 35;  date cols → dozens of useless ones.
    HIGH_CARDINALITY = [
        "addr_state", "sub_grade", "emp_title", "url", "desc", "title",
        "zip_code", "earliest_cr_line", "last_pymnt_d", "next_pymnt_d",
        "last_credit_pull_d", "issue_d",
    ]
    drop_cols = [c for c in HIGH_CARDINALITY if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        log.info("Dropped high-cardinality columns: %s", drop_cols)

    df = _load_alternative_data(df)
    df = create_features(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X = pd.get_dummies(X, drop_first=True)
    X.columns = sanitize_columns(X.columns)
    X = X.astype("float32")
    log.info("After encoding: %d features", X.shape[1])

    # Feature selection: top-80 by XGBoost importance reduces noise from 800+ cols
    MAX_FEATURES = 80
    if X.shape[1] > MAX_FEATURES:
        log.info("Feature selection: %d → top %d …", X.shape[1], MAX_FEATURES)
        X_sub, _, y_sub, _ = train_test_split(
            X, y, test_size=0.5, random_state=RANDOM_STATE, stratify=y
        )
        selector = XGBClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", tree_method="hist",
            random_state=RANDOM_STATE, device="cpu",
        )
        selector.fit(X_sub, y_sub)
        importances  = pd.Series(selector.feature_importances_, index=X.columns)
        top_features = importances.nlargest(MAX_FEATURES).index.tolist()
        X = X[top_features]
        log.info("Feature selection complete → %d features retained", len(top_features))

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 2. SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split(X, y):
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN — XGBoost only
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train) -> XGBClassifier:
    """
    Train a single XGBoost model with GridSearchCV.

    SMOTE is applied on the training split ONLY to avoid data leakage.
    SMOTE is capped at 30 000 rows to prevent multi-minute runtimes on
    large datasets (still representative due to stratified sampling).
    """
    SMOTE_CAP = 30_000
    counter_orig = Counter(y_train)
    log.info("Class distribution before SMOTE: %s", dict(counter_orig))

    try:
        from imblearn.over_sampling import SMOTE, RandomOverSampler

        n_minority = counter_orig.get(1, 0)

        if n_minority < 6:
            log.warning("Minority class too small (%d) — using RandomOverSampler", n_minority)
            ros = RandomOverSampler(random_state=RANDOM_STATE)
            X_res, y_res = ros.fit_resample(X_train, y_train)
        elif len(X_train) > SMOTE_CAP:
            log.info("Dataset (%d rows) exceeds SMOTE cap (%d). Subsampling first.", len(X_train), SMOTE_CAP)
            cap_ratio = SMOTE_CAP / len(X_train)
            sub_idx = (
                pd.Series(y_train.values)
                .groupby(y_train.values)
                .apply(lambda g: g.sample(frac=cap_ratio, random_state=RANDOM_STATE))
                .index.get_level_values(1)
            )
            X_sub, y_sub = X_train.iloc[sub_idx], y_train.iloc[sub_idx]
            smote = SMOTE(random_state=RANDOM_STATE,
                          k_neighbors=min(5, Counter(y_sub)[1] - 1))
            X_sm, y_sm = smote.fit_resample(X_sub, y_sub)
            log.info("SMOTE on subsample: %d → %d", len(X_sub), len(X_sm))
            # Combine synthetic minority samples with the full original training set
            synthetic_only = pd.DataFrame(X_sm, columns=X_train.columns).iloc[len(X_sub):]
            y_synthetic    = pd.Series(y_sm).iloc[len(y_sub):]
            X_res = pd.concat([X_train, synthetic_only], ignore_index=True)
            y_res = pd.concat([y_train, y_synthetic],   ignore_index=True)
            log.info("Combined training set: %d rows", len(X_res))
        else:
            smote = SMOTE(random_state=RANDOM_STATE,
                          k_neighbors=min(5, n_minority - 1))
            X_res, y_res = smote.fit_resample(X_train, y_train)
            log.info("SMOTE applied: %d → %d", len(X_train), len(X_res))

    except ImportError:
        log.warning("imbalanced-learn not installed — training without SMOTE.")
        X_res, y_res = X_train, y_train

    counter      = Counter(y_res)
    scale_pos_wt = counter.get(0, 1) / max(counter.get(1, 1), 1)

    xgb_base = XGBClassifier(
        scale_pos_weight = scale_pos_wt,
        eval_metric      = "aucpr",
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 3,
        random_state     = RANDOM_STATE,
        tree_method      = "hist",
        device           = "cpu",
    )

    param_grid = {
        "n_estimators":  [100, 200, 300],
        "max_depth":     [4, 6],
    }

    grid_search = GridSearchCV(
        estimator  = xgb_base,
        param_grid = param_grid,
        scoring    = "roc_auc",
        cv         = 3,
        n_jobs     = -1,
        verbose    = 1,
    )
    grid_search.fit(X_res, y_res)
    log.info("Best XGBoost params: %s", grid_search.best_params_)
    log.info("Best ROC-AUC (CV):   %.4f", grid_search.best_score_)
    return grid_search.best_estimator_


# ─────────────────────────────────────────────────────────────────────────────
# 4. EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model: XGBClassifier, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    # Optimal decision threshold via Youden's J (maximise TPR − FPR)
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    best_threshold = float(thresholds[(tpr - fpr).argmax()])

    recall  = recall_score(y_test, preds,  zero_division=0)
    f1      = f1_score(y_test, preds,      zero_division=0)
    roc_auc = roc_auc_score(y_test, probs)

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    metrics = {
        "model_name":         "xgboost",
        "accuracy":           round(float(accuracy_score(y_test, preds)),                   4),
        "precision":          round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall":             round(float(recall),                                           4),
        "f1_score":           round(float(f1),                                               4),
        "roc_auc":            round(float(roc_auc),                                          4),
        "decision_threshold": round(best_threshold,                                          6),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
        },
    }

    log.info(
        "XGBoost — recall=%.4f  f1=%.4f  roc_auc=%.4f  threshold=%.4f",
        metrics["recall"], metrics["f1_score"],
        metrics["roc_auc"], best_threshold,
    )
    log.info("\n%s", classification_report(y_test, preds))
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 5. SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_artifacts(model: XGBClassifier, metrics: dict, feature_names: list) -> None:
    import pickle

    os.makedirs(os.path.dirname(CHAMPION_MODEL_PATH), exist_ok=True)
    joblib.dump(model, CHAMPION_MODEL_PATH)
    log.info("Champion model saved → %s", CHAMPION_MODEL_PATH)

    os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(feature_names, f)
    log.info("Feature list saved  → %s", FEATURES_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    log.info("Metrics saved       → %s", METRICS_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train AegisBank XGBoost Champion Model")
    parser.add_argument("--push-to-hf", action="store_true", help="Push trained model artifacts to Hugging Face Hub")
    parser.add_argument("--hf-repo", type=str, default=None, help="Hugging Face model repository ID")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face Access Token")

    args, _ = parser.parse_known_args()

    X, y                             = load_and_preprocess()
    X_train, X_test, y_train, y_test = split(X, y)
    model                            = train_xgboost(X_train, y_train)
    metrics                          = evaluate(model, X_test, y_test)
    save_artifacts(model, metrics, list(X.columns))
    log.info("Training pipeline complete ✅")

    if args.push_to_hf or os.environ.get("HF_AUTO_PUSH", "false").lower() in ("true", "1", "yes"):
        log.info("Pushing model artifacts to Hugging Face Hub...")
        try:
            from utils.hf_hub import upload_model_to_hf
            res = upload_model_to_hf(repo_id=args.hf_repo, token=args.hf_token)
            log.info("Hugging Face upload response: %s", res)
        except Exception as e:
            log.error("Failed to push model to Hugging Face Hub: %s", e)


if __name__ == "__main__":
    main()

