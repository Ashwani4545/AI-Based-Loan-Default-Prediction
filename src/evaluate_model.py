# src/evaluate_model.py

import pandas as pd
import joblib

from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report

from utils.config import PROCESSED_DATA_PATH, MODEL_PATH, TARGET_COLUMN


def load_model():
    return joblib.load(MODEL_PATH)


def load_data():
    return pd.read_csv(PROCESSED_DATA_PATH)


def evaluate():

    model = load_model()
    data = load_data()

    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, preds))

    print("\nClassification Report:")
    print(classification_report(y, preds))

    print("\nROC-AUC Score:")
    print(roc_auc_score(y, probs))


if __name__ == "__main__":
    evaluate()