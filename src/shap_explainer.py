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
            seen[c] += 1
            c = f"{c}_{seen[c]}"
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
            # Use TreeExplainer for XGBoost — exact, fast.
            # Fall back to generic Explainer for other model types.
            # If both fail, defer initialisation until data is available.
            try:
                self.explainer = self.shap.TreeExplainer(self.model)
                log.info("SHAP TreeExplainer initialised ✅ (fast path)")
            except Exception:
                try:
                    self.explainer = self.shap.Explainer(self.model)
                    log.info("SHAP generic Explainer initialised ✅ (fallback)")
                except Exception:
                    self.explainer = None
                    log.info("SHAP explainer initialisation deferred until data is available.")
        else:
            self.explainer = None
            log.warning("SHAP is not installed; using feature-importance fallback for explanations.")

    def reload(self, model_path: str = None) -> None:
        """Reload model from disk after retraining and reinitialise SHAP explainer."""
        path = model_path or MODEL_PATH
        self.model = joblib.load(path)
        if self.has_shap:
            try:
                self.explainer = self.shap.TreeExplainer(self.model)
                log.info("SHAP TreeExplainer reloaded ✅ (fast path)")
            except Exception:
                try:
                    self.explainer = self.shap.Explainer(self.model)
                    log.info("SHAP generic Explainer reloaded ✅ (fallback)")
                except Exception:
                    self.explainer = None
                    log.info("SHAP explainer initialisation deferred until data is available.")

    def _fallback_importances(self, columns: pd.Index) -> np.ndarray:
        """Return feature importances when SHAP is unavailable."""
        try:
            if hasattr(self.model, "feature_importances_"):
                importances = np.asarray(self.model.feature_importances_, dtype=float)
                if len(importances) == len(columns):
                    return importances
            if hasattr(self.model, "coef_"):
                importances = np.abs(np.asarray(self.model.coef_)[0])
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

        # Lazy initialisation: if explainer was deferred, initialise with background data now
        if self.explainer is None:
            try:
                log.info("Initialising generic Explainer with background data...")
                background = self.shap.maskers.Independent(X, max_samples=100)
                self.explainer = self.shap.Explainer(self.model, background)
                log.info("SHAP Explainer initialised with background data ✅")
            except Exception as e:
                log.warning("Could not initialise explainer with background data: %s", e)
                self.has_shap = False
                return None

        log.info("Computing SHAP values for %d samples …", len(X))
        return self.explainer(X)

    def explain_single(self, input_df: pd.DataFrame) -> list:
        """
        Return top-5 SHAP feature drivers for a single prediction row.

        Each entry: {"feature": str, "shap_value": float, "impact": float}
        - shap_value: signed (positive = pushes toward default, negative = toward repay)
        - impact: absolute magnitude (used for bar width)
        """
        raw_values: np.ndarray

        if self.has_shap and self.explainer is not None:
            try:
                sv = self.explainer(input_df)
                raw_values = sv.values[0]
            except Exception:
                log.exception("SHAP explain_single failed — using fallback importances")
                raw_values = self._fallback_importances(input_df.columns)
        else:
            raw_values = self._fallback_importances(input_df.columns)

        feature_names = input_df.columns
        # Sort by absolute magnitude descending, take top 5
        top_idx = np.abs(raw_values).argsort()[-5:][::-1]

        explanation = []
        for i in top_idx:
            sv = float(raw_values[i])
            explanation.append({
                "feature":    str(feature_names[i]),
                "shap_value": round(sv, 6),
                "impact":     round(abs(sv), 6),
                # direction: True = increases default risk, False = decreases it
                "increases_risk": sv > 0,
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

    def check_individual_fairness(self, input_data: dict) -> str:
        """Simple financial-ratio fairness heuristic."""
        income = float(input_data.get("annual_inc", 0) or 0)
        loan   = float(input_data.get("loan_amnt",  0) or 0)
        ratio  = loan / (income + 1e-6)
        if ratio > 5:
            return "⚠️ High financial risk ratio"
        return "✅ No obvious bias pattern"

    def check_group_bias(self, input_data: dict) -> str:
        """
        Detect potential financially-vulnerable applicant patterns.
        Gender, race, and religion are never collected in the form —
        this check uses financial proxies only.
        """
        income = float(input_data.get("annual_inc",     0) or 0)
        loan   = float(input_data.get("loan_amnt",      0) or 0)
        fico   = float(input_data.get("fico_range_low", 0) or 0)

        if income < 30_000 and fico == 0:
            return "⚠️ Potentially credit-invisible low-income applicant — alternative data used"
        if income > 0 and loan / income > 4:
            return "⚠️ High loan-to-income ratio — elevated risk for financial distress"
        return "✅ No bias pattern detected"

    def validate_sensitive_features(self, input_data: dict) -> list:
        """
        Warn if prohibited sensitive attributes appear in the payload.
        NOTE: gender, race, religion are never collected in the form,
        so this fires only if the REST API is called with extra fields.
        """
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

        y_pred = self.predict(X)

        # Subsample for SHAP to avoid OOM errors on large datasets
        X_shap = X.sample(n=min(1000, len(X)), random_state=42) if len(X) > 1000 else X
        shap_values = self.generate_shap_values(X_shap)

        self.save_summary_plot(shap_values, X_shap, output_dir)
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
