import pandas as pd
import json
from pathlib import Path

from utils.config import HISTORY_PATH, PROCESSED_DATA_PATH

def build_feedback_dataset():
    try:
        with open(HISTORY_PATH) as f:
            history = json.load(f)
    except Exception:
        return None

    if len(history) < 100:
        return None

    df = pd.DataFrame(history)

    # Convert prediction labels to numeric if needed
    if "prediction" in df.columns and df["prediction"].dtype == object:
        mapping = {"Repay": 0, "Default": 1}
        df["prediction"] = df["prediction"].map(mapping)

<<<<<<< HEAD
    # FIX: rename first so the column name matches what we select below
    df = df.rename(columns={"fico": "fico_range_low"})

    # Extract features (using fico_range_low after rename)
=======
    # Extract features — Bug #12 fix: use fico_range_low directly
>>>>>>> 44ab82bb832d0cf735042468c185eb3463bf6a67
    features = [
        "loan_amnt", "int_rate", "annual_inc", "fico_range_low",
        "dti", "open_acc", "revol_bal", "total_acc"
    ]

<<<<<<< HEAD
    df = df[features + ["prediction"]].dropna()
=======
    # Rename legacy "fico" column if present
    if "fico" in df.columns and "fico_range_low" not in df.columns:
        df = df.rename(columns={"fico": "fico_range_low"})

    available = [c for c in features + ["prediction"] if c in df.columns]
    df = df[available].dropna()
>>>>>>> 44ab82bb832d0cf735042468c185eb3463bf6a67

    # Rename prediction → target
    df = df.rename(columns={"prediction": "loan_status"})
    return df


def update_training_data(feedback_df: pd.DataFrame) -> bool:
    if feedback_df is None or feedback_df.empty:
        return False

    try:
        if Path(PROCESSED_DATA_PATH).exists():
            base_df = pd.read_csv(PROCESSED_DATA_PATH)
        else:
            base_df = pd.DataFrame()

        combined = pd.concat([base_df, feedback_df], ignore_index=True)
        combined.to_csv(PROCESSED_DATA_PATH, index=False)
        return True
    except Exception:
        return False