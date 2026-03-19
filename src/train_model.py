# src/train_model.py

import sys
import os
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from utils.config import (
    PROCESSED_DATA_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_PATH,
    XGB_PARAMS,
)


# ========================
# LOAD DATA
# ========================

def load_data():
    data = pd.read_csv(PROCESSED_DATA_PATH)
    return data


# ========================
# PREPROCESS DATA
# ========================

def _sanitize_and_uniquify_columns(columns):
    seen = {}
    cleaned = []

    for col in columns:
        c = str(col)
        c = re.sub(r"[\[\]<>]", "_", c)      # xgboost-forbidden chars
        c = re.sub(r"\s+", "_", c.strip())   # spaces -> _
        c = re.sub(r"[^0-9a-zA-Z_]", "_", c) # keep safe chars

        if c in seen:
            seen[c] += 1
            c = f"{c}_{seen[c]}"
        else:
            seen[c] = 0

        cleaned.append(c)

    return cleaned


def preprocess_data(data):
    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]
    
    # Convert object/string columns to numeric
    X = pd.get_dummies(X, drop_first=True)

    # Ensure XGBoost-compatible feature names
    X.columns = _sanitize_and_uniquify_columns(X.columns)

    # Ensure numeric dtype for all models
    X = X.astype("float32")
    
    return X, y



# ========================
# SPLIT DATA
# ========================

def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


# ========================
# TRAIN MODELS
# ========================

def train_models(X_train, y_train):

    models = {
        "logistic_regression": LogisticRegression(max_iter=5000, solver="saga"),
        "random_forest": RandomForestClassifier(n_estimators=100),
        "xgboost": XGBClassifier(**XGB_PARAMS)
    }

    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


# ========================
# EVALUATE MODELS
# ========================

def evaluate_models(models, X_test, y_test):

    results = {}

    for name, model in models.items():

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print(f"\nModel: {name}")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds))

        results[name] = acc

    return results


# ========================
# SAVE BEST MODEL
# ========================

def save_best_model(models, results):

    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]

    print(f"\nBest Model: {best_model_name}")

    joblib.dump(best_model, MODEL_PATH)

    print(f"Model saved at: {MODEL_PATH}")


# ========================
# MAIN
# ========================

def main():

    data = load_data()

    X, y = preprocess_data(data)

    X_train, X_test, y_train, y_test = split_data(X, y)

    models = train_models(X_train, y_train)

    results = evaluate_models(models, X_test, y_test)

    save_best_model(models, results)


if __name__ == "__main__":
    main()