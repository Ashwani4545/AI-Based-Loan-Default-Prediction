# src/shap_explainer.py
"""
Loan Default Prediction — SHAP Explainability & Fairness

Generates:
  - SHAP summary plot  (shap_summary.png)
  - SHAP force plot    (shap_force_plot.html)
  - Fairness report    (fairness_report.txt)

Usage:
    python -m src.shap_explainer
"""

import os
import re
import sys
import logging
import importlib
import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import MODEL_PATH, PROCESSED_DATA_PATH, TARGET_COLUMN, SENSITIVE_COLUMN

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN SANITIZER  (must match train_model.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize_columns(columns) -> list:
    seen: dict = {}
    result: list = []
    for col in columns:
        c = re.sub(r"[\[\]<>]", "_", str(col))
        c = re.sub(r"\s+",      "_", c.strip())
        c = re.sub(r"[^0-9a-zA-Z_]", "_", c)
        if c in seen:
            seen[c] += 1; c = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
        result.append(c)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# EXPLAINER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class LoanModelExplainer:

    def __init__(self, model_path: str = MODEL_PATH):
        self.model = joblib.load(model_path)

        spec = importlib.util.find_spec("shap")
        self.has_shap = spec is not None
        self.shap = importlib.import_module("shap") if self.has_shap else None

        if self.has_shap:
            self.explainer = self.shap.Explainer(self.model)
            log.info("SHAP explainer initialised ✅")
        else:
            self.explainer = None
            log.warning("SHAP is not installed; using feature-importance fallback for explanations.")

    def reload(self, model_path: str = None):
        """Reload model from disk (call after retraining) — Bug #14 fix."""
        path = model_path or MODEL_PATH
        self.model = joblib.load(path)
        if self.has_shap:
            self.explainer = self.shap.Explainer(self.model)
        log.info("SHAP explainer reloaded ✅")

    def _fallback_importances(self, columns: pd.Index) -> np.ndarray:
        try:
            if hasattr(self.model, "feature_importances_"):
                importances = np.asarray(self.model.feature_importances_, dtype=float)
                if len(importances) == len(columns):
                    return importances
            if hasattr(self.model, "get_booster"):
                booster = self.model.get_booster()
                score_map = booster.get_score(importance_type="weight")
                return np.asarray([float(score_map.get(col, 0.0)) for col in columns], dtype=float)
        except Exception:
            pass

        return np.zeros(len(columns), dtype=float)

    # ── DATA LOADING ─────────────────────────────────────────────────────────

    def load_data(self, file_path: str, target_column: str):
        df = pd.read_csv(file_path)

        if target_column not in df.columns:
            raise KeyError(f"Target '{target_column}' not found. Columns: {df.columns.tolist()}")

        # Preserve raw df for sensitive attribute lookup before encoding
        raw_df = df.copy()

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X = pd.get_dummies(X, drop_first=True)
        X.columns = _sanitize_columns(X.columns)
        X = X.astype("float32")

        # Align to model features
        try:
            model_features = self.model.get_booster().feature_names
        except AttributeError:
            model_features = list(getattr(self.model, "feature_names_in_", X.columns))

        for col in model_features:
            if col not in X.columns:
                X[col] = 0.0
        X = X[model_features]

        return raw_df, X, y

    # ── PREDICTION ───────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)

    # ── SHAP ─────────────────────────────────────────────────────────────────

    def generate_shap_values(self, X: pd.DataFrame):
        if not self.has_shap:
            log.info("SHAP unavailable; returning no SHAP values.")
            return None

        log.info("Computing SHAP values for %d samples …", len(X))
        return self.explainer(X)

    def explain_single(self, input_df: pd.DataFrame):
        if self.has_shap:
            shap_values = self.explainer(input_df)
            importance = np.abs(shap_values.values[0])
        else:
            importance = np.abs(self._fallback_importances(input_df.columns))

        # Get top 5 important features
        feature_names = input_df.columns
        
        top_idx = importance.argsort()[-5:][::-1]
        
        explanation = []
        for i in top_idx:
            explanation.append({
                "feature": feature_names[i],
                "impact": round(float(importance[i]), 4)
            })
        
        return explanation

    def save_summary_plot(self, shap_values, X: pd.DataFrame, output_dir: str) -> None:
        if not self.has_shap or shap_values is None:
            log.info("Skipping SHAP summary plot because SHAP is unavailable.")
            return

        plt.figure(figsize=(10, 6))
        self.shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        path = os.path.join(output_dir, "shap_summary.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("SHAP summary plot → %s", path)

    def save_force_plot(self, shap_values, index: int, output_dir: str) -> None:
        if not self.has_shap or shap_values is None:
            log.info("Skipping SHAP force plot because SHAP is unavailable.")
            return

        force = self.shap.plots.force(shap_values[index])
        path  = os.path.join(output_dir, "shap_force_plot.html")
        self.shap.save_html(path, force)
        log.info("SHAP force plot  → %s", path)

    # ── FAIRNESS ─────────────────────────────────────────────────────────────

    def demographic_parity(self, y_pred, sensitive_attr: pd.Series) -> pd.Series:
        """Mean prediction rate per group."""
        df = pd.DataFrame({"prediction": y_pred, "group": sensitive_attr.values})
        return df.groupby("group")["prediction"].mean()

    def equal_opportunity(self, y_true, y_pred, sensitive_attr: pd.Series) -> dict:
        """True-Positive Rate (recall) per group."""
        df = pd.DataFrame({
            "y_true": y_true.values,
            "y_pred": y_pred,
            "group":  sensitive_attr.values,
        })
        results: dict = {}
        for group, gdf in df.groupby("group"):
            if gdf["y_true"].nunique() < 2:
                results[group] = 0.0
                continue
            tn, fp, fn, tp = confusion_matrix(gdf["y_true"], gdf["y_pred"]).ravel()
            results[group] = round(tp / (tp + fn + 1e-9), 4)
        return results

    def check_individual_fairness(self, input_data: dict):
        # Simple fairness heuristic
        income = float(input_data.get("annual_inc", 0))
        loan   = float(input_data.get("loan_amnt", 0))
        
        ratio = loan / (income + 1e-6)
        
        if ratio > 5:
            return "⚠️ High financial risk ratio"
        
        return "✅ No obvious bias pattern"

    def check_group_bias(self, input_data: dict):
        """
        Check bias using available attributes (income, state, etc.).
        Bug #10 fix: removed 'gender' check — field is never in the form.
        """
        income = float(input_data.get("annual_inc", 0) or 0)
        state = input_data.get("addr_state", "")

        # Flag low-income borrowers for bias review
        if income < 20000 and state:
            return "⚠️ Potential bias risk: low-income borrower"

        return "✅ No bias detected"

    def validate_sensitive_features(self, input_data: dict):
        # NOTE: gender, race, religion are never collected in the form,
        # so this check is informational only for API usage.
        sensitive_fields = ["gender", "race", "religion"]

        warnings = []
        for field in sensitive_fields:
            if field in input_data:
                warnings.append(f"{field} should not influence decision")

        return warnings

    # ── FULL REPORT ──────────────────────────────────────────────────────────

    def generate_reports(
        self,
        data_path:        str = PROCESSED_DATA_PATH,
        target_column:    str = TARGET_COLUMN,
        sensitive_column: str = SENSITIVE_COLUMN,
        output_dir:       str = "outputs",
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)

        raw_df, X, y = self.load_data(data_path, target_column)

        # Check sensitive column exists in raw (pre-encoding) df
        if sensitive_column not in raw_df.columns:
            log.warning("Sensitive column '%s' not found; skipping fairness metrics.", sensitive_column)
            sensitive_col = None
        else:
            sensitive_col = raw_df[sensitive_column]

        y_pred      = self.predict(X)
        shap_values = self.generate_shap_values(X)

        self.save_summary_plot(shap_values, X, output_dir)
        self.save_force_plot(shap_values, index=0, output_dir=output_dir)

        # Fairness report
        report_path = os.path.join(output_dir, "fairness_report.txt")
        with open(report_path, "w") as f:
            if sensitive_col is not None:
                dp = self.demographic_parity(y_pred, sensitive_col)
                eo = self.equal_opportunity(y, y_pred, sensitive_col)
                f.write("=== Demographic Parity (avg prediction rate per group) ===\n")
                f.write(dp.to_string() + "\n\n")
                f.write("=== Equal Opportunity (TPR per group) ===\n")
                for group, tpr in eo.items():
                    f.write(f"  {group}: {tpr:.4f}\n")
            else:
                f.write("Fairness metrics skipped — sensitive column not available.\n")

        log.info("Fairness report → %s", report_path)
        log.info("All reports generated ✅")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    explainer = LoanModelExplainer(str(MODEL_PATH))
    explainer.generate_reports(
        data_path=str(PROCESSED_DATA_PATH),
        target_column=TARGET_COLUMN,
        sensitive_column=SENSITIVE_COLUMN,
        output_dir=str(base / "outputs"),
    )
