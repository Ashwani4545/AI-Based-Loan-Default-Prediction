# src/evaluate_model.py

import sys
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, accuracy_score, precision_score, recall_score, f1_score
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import PROCESSED_DATA_PATH, MODEL_PATH, TARGET_COLUMN


def load_model():
    return joblib.load(MODEL_PATH)


def load_data():
    return pd.read_csv(PROCESSED_DATA_PATH)


def preprocess_data(X):
    """Apply same preprocessing as training"""
    # One-hot encode categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        print(f"🔄 One-hot encoding {len(categorical_cols)} categorical columns...")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    
    # Ensure all columns are numeric
    X = X.astype(np.float32)
    return X


def evaluate():
    model = load_model()
    data = load_data()

    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]

    print(f"📊 Original shape: {X.shape}")
    
    # Preprocess (one-hot encode)
    X = preprocess_data(X)
    print(f"✅ After preprocessing: {X.shape}")
    
    # Get model features
    booster = model.get_booster()
    model_features = booster.feature_names
    print(f"🎯 Model expects {len(model_features)} features")
    
    # Align columns: add missing columns, remove extra columns
    for col in model_features:
        if col not in X.columns:
            X[col] = 0.0
    
    X = X[model_features].astype(np.float32)
    print(f"✅ Aligned to model features: {X.shape}")

    # Make predictions
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    cm = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y, preds)
    precision = precision_score(y, preds)
    recall = recall_score(y, preds)
    f1 = f1_score(y, preds)
    roc_auc = roc_auc_score(y, probs)

    print("\n" + "="*50)
    print("MODEL EVALUATION METRICS")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(f"TN: {tn}, FP: {fp}")
    print(f"FN: {fn}, TP: {tp}")

    print("\nClassification Report:")
    print(classification_report(y, preds))

    # Save metrics to JSON
    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }

    metrics_path = Path(__file__).resolve().parent.parent / "model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\n✅ Metrics saved to {metrics_path}")


if __name__ == "__main__":
    evaluate()