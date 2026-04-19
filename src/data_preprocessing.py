# src/data_preprocessing.py
"""
Loan Default Prediction — Data Preprocessing & Feature Engineering

Pipeline:
  1. Load raw CSV
  2. Clean (dedup, fill nulls)
  3. Engineer domain features
  4. Encode categoricals
  5. Split → Scale → SMOTE
  6. Save processed data
"""

import os
import sys
import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── ensure project root on path ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ── OPTIONAL DEPENDENCY (imblearn) ────────────────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    log.warning("imbalanced-learn not installed — SMOTE step will be skipped.")


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    log.info("Loaded dataset: %s rows × %s cols", *df.shape)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. CLEAN
# ─────────────────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    log.info("Dropped %d duplicate rows", before - len(df))

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    log.info("Nulls filled — numeric: median, categorical: mode")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-specific derived features (safe: only if source cols exist)."""

    def safe_ratio(num_col, den_col, new_col):
        if num_col in df.columns and den_col in df.columns:
            df[new_col] = df[num_col] / (df[den_col].replace(0, np.nan) + 1)

    # LendingClub-style derived features
    safe_ratio("loan_amnt",    "annual_inc",   "loan_income_ratio")
    safe_ratio("revol_bal",    "annual_inc",   "revol_income_ratio")
    safe_ratio("open_acc",     "total_acc",    "open_acc_ratio")
    safe_ratio("installment",  "annual_inc",   "installment_income_ratio")

    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

    if "int_rate" in df.columns and "dti" in df.columns:
        df["risk_score"] = df["int_rate"] * df["dti"]

    log.info("Feature engineering complete")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. ENCODE
# ─────────────────────────────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode all object columns (drop_first to avoid multicollinearity)."""
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        log.info("One-hot encoded %d categorical columns", len(cat_cols))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. SPLIT + SCALE
# ─────────────────────────────────────────────────────────────────────────────

def split_and_scale(df: pd.DataFrame):
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not in dataframe.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    log.info("Train: %s  |  Test: %s", X_train.shape, X_test.shape)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Handle class imbalance
    if HAS_SMOTE:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_res, y_train_res = smote.fit_resample(X_train_sc, y_train)
        log.info("SMOTE applied — resampled shape: %s", X_train_res.shape)
    else:
        X_train_res, y_train_res = X_train_sc, y_train
        log.info("SMOTE skipped — using raw split")

    return X_train_res, X_test_sc, y_train_res, y_test, X.columns.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# 6. SAVE PROCESSED DATA
# ─────────────────────────────────────────────────────────────────────────────

def save_processed(df: pd.DataFrame, path: str = PROCESSED_DATA_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Processed data saved → %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing() -> None:
    df = load_raw_data()
    df = clean_data(df)
    df = engineer_features(df)

    # Fill any NaN values introduced by feature engineering (before scaling/SMOTE)
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Save lightly-cleaned version for model training (keep target)
    save_processed(df)

    # Full pipeline report
    df_enc = encode_categoricals(df.copy())
    X_train, X_test, y_train, y_test, features = split_and_scale(df_enc)

    log.info("Preprocessing complete.")
    log.info("Final train shape: %s | test shape: %s", X_train.shape, X_test.shape)


if __name__ == "__main__":
    run_preprocessing()