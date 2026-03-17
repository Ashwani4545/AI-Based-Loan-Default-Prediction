# utils/config.py

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ========================
# DATA PATHS
# ========================

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "loan_data.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")

# ========================
# MODEL PATH
# ========================

MODEL_PATH = os.path.join(BASE_DIR, "models", "loan_default_model.pkl")

# ========================
# TARGET VARIABLE
# ========================

TARGET_COLUMN = "loan_status"   # Change if your column name differs

# ========================
# TRAINING PARAMETERS
# ========================

TEST_SIZE = 0.2
RANDOM_STATE = 42

# ========================
# MODEL PARAMETERS
# ========================

XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE
}