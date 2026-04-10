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

    # Extract features
    features = [
        "loan_amnt", "int_rate", "annual_inc", "fico",
        "dti", "open_acc", "revol_bal", "total_acc"
    ]

    df = df.rename(columns={"fico": "fico_range_low"})

    df = df[features + ["prediction"]].dropna()

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