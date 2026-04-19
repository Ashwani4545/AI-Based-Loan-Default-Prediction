"""
Generate standard ML figures for the loan default project.

Figures:
1) Before/After SMOTE class distribution
2) ROC-AUC + Precision-Recall curves
3) SHAP summary plot

Usage examples:
  python -m src.generate_performance_plots --plot smote
  python -m src.generate_performance_plots --plot curves
  python -m src.generate_performance_plots --plot shap
  python -m src.generate_performance_plots --plot all
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from utils.config import PROCESSED_DATA_PATH, RANDOM_STATE, TARGET_COLUMN


def sanitize_columns(columns) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for col in columns:
        c = str(col).strip().replace(" ", "_").replace("[", "_").replace("]", "_")
        c = c.replace("<", "_").replace(">", "_")
        c = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in c)
        if c in seen:
            seen[c] += 1
            c = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
        out.append(c)
    return out


def coerce_target_binary(y: pd.Series) -> pd.Series:
    if y.dropna().isin([0, 1]).all():
        return y.astype(int)

    y_str = y.astype(str).str.lower().str.strip()
    mapping = {
        "fully paid": 0,
        "current": 0,
        "non-default": 0,
        "repaid": 0,
        "charged off": 1,
        "default": 1,
        "late": 1,
    }
    mapped = y_str.map(mapping)
    if mapped.isna().any():
        raise ValueError(
            "Target column is not binary and contains unknown labels. "
            "Please map labels in coerce_target_binary()."
        )
    return mapped.astype(int)


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(PROCESSED_DATA_PATH)
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in processed data.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = coerce_target_binary(df[TARGET_COLUMN])

    X = pd.get_dummies(X, drop_first=True)
    X.columns = sanitize_columns(X.columns)
    return X.astype("float32"), y


def train_baseline_model(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[XGBClassifier, pd.DataFrame, pd.Series]:
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_sm, y_train_sm)
    return model, X_train_sm, y_train_sm


def plot_smote_distribution(y_train: pd.Series, y_train_sm: pd.Series, output_dir: Path) -> Path:
    before = y_train.value_counts().sort_index()
    after = y_train_sm.value_counts().sort_index()

    labels = ["Repay (0)", "Default (1)"]
    before_vals = [int(before.get(0, 0)), int(before.get(1, 0))]
    after_vals = [int(after.get(0, 0)), int(after.get(1, 0))]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)

    axes[0].bar(labels, before_vals, color=["#1f77b4", "#d62728"])
    axes[0].set_title("Before SMOTE")
    axes[0].set_ylabel("Count")

    axes[1].bar(labels, after_vals, color=["#1f77b4", "#d62728"])
    axes[1].set_title("After SMOTE")

    fig.suptitle("Class Distribution: Original vs SMOTE-Resampled")
    fig.tight_layout()

    out_path = output_dir / "smote_class_distribution.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_roc_pr_curves(y_test: pd.Series, y_prob: pd.Series, output_dir: Path) -> Path:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")

    axes[1].plot(recall, precision, color="#2ca02c", linewidth=2, label=f"AP = {ap:.3f}")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")

    fig.suptitle("XGBoost Performance Curves")
    fig.tight_layout()

    out_path = output_dir / "roc_pr_curves.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_shap_summary(model: XGBClassifier, X_sample: pd.DataFrame, output_dir: Path) -> Path:
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()

    out_path = output_dir / "shap_summary_plot.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return out_path


def main(plot: str) -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model, X_train_sm, y_train_sm = train_baseline_model(X_train, y_train)
    y_prob = pd.Series(model.predict_proba(X_test)[:, 1], index=y_test.index)

    generated: list[Path] = []

    if plot in {"smote", "all"}:
        generated.append(plot_smote_distribution(y_train, y_train_sm, output_dir))

    if plot in {"curves", "all"}:
        generated.append(plot_roc_pr_curves(y_test, y_prob, output_dir))

    if plot in {"shap", "all"}:
        shap_sample = X_test.sample(min(1000, len(X_test)), random_state=RANDOM_STATE)
        generated.append(plot_shap_summary(model, shap_sample, output_dir))

    print("Generated files:")
    for p in generated:
        print(f"- {os.fspath(p)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate model performance and explainability plots.")
    parser.add_argument(
        "--plot",
        choices=["smote", "curves", "shap", "all"],
        default="all",
        help="Choose which plot(s) to generate.",
    )
    args = parser.parse_args()
    main(args.plot)
