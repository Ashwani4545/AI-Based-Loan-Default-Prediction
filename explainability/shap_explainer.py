# shap_explainer.py

import os
import importlib
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path
import re


class LoanModelExplainer:
    def __init__(self, model_path):
        """
        Initialize model and SHAP explainer
        """
        self.model = joblib.load(model_path)

        shap_spec = importlib.util.find_spec("shap")
        if shap_spec is None:
            raise ImportError("The 'shap' package is not installed. Install it with: pip install shap")

        self.shap = importlib.import_module("shap")
        self.explainer = self.shap.Explainer(self.model)

    # =========================
    # COLUMN SANITIZATION
    # =========================
    def _sanitize_and_uniquify_columns(self, columns):
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

    # =========================
    # DATA LOADING
    # =========================
    def load_data(self, file_path, target_column):
        df = pd.read_csv(file_path)
        
        print(f"Available columns: {df.columns.tolist()}")
        print(f"Looking for target: {target_column}")
        
        if target_column not in df.columns:
            raise KeyError(f"Target column '{target_column}' not found. Available: {df.columns.tolist()}")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Apply same preprocessing as training
        X = pd.get_dummies(X, drop_first=True)
        X.columns = self._sanitize_and_uniquify_columns(X.columns)
        X = X.astype("float32")
        
        return df, X, y

    # =========================
    # PREDICTION
    # =========================
    def predict(self, X):
        return self.model.predict(X)

    # =========================
    # SHAP EXPLAINABILITY
    # =========================
    def generate_shap_values(self, X):
        return self.explainer(X)

    def save_summary_plot(self, shap_values, X, output_path):
        plt.figure()
        self.shap.summary_plot(shap_values, X, show=False)
        plt.savefig(os.path.join(output_path, "shap_summary.png"))
        plt.close()

    def save_force_plot(self, shap_values, index, output_path):
        force = self.shap.plots.force(shap_values[index])
        self.shap.save_html(os.path.join(output_path, "shap_force_plot.html"), force)

    # =========================
    # FAIRNESS METRICS
    # =========================
    def demographic_parity(self, y_pred, sensitive_attr):
        df = pd.DataFrame({
            "prediction": y_pred,
            "group": sensitive_attr
        })
        return df.groupby("group")["prediction"].mean()

    def equal_opportunity(self, y_true, y_pred, sensitive_attr):
        df = pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
            "group": sensitive_attr
        })

        results = {}

        for group in df["group"].unique():
            group_df = df[df["group"] == group]

            # Handle edge case
            if len(group_df["y_true"].unique()) < 2:
                results[group] = 0
                continue

            tn, fp, fn, tp = confusion_matrix(
                group_df["y_true"], group_df["y_pred"]
            ).ravel()

            tpr = tp / (tp + fn + 1e-6)
            results[group] = tpr

        return results

    # =========================
    # REPORT GENERATION
    # =========================
    def generate_reports(self, data_path, target_column, sensitive_column, output_path):
        os.makedirs(output_path, exist_ok=True)

        # Load data
        df, X, y = self.load_data(data_path, target_column)

        # Predictions
        y_pred = self.predict(X)

        # SHAP
        shap_values = self.generate_shap_values(X)

        self.save_summary_plot(shap_values, X, output_path)
        self.save_force_plot(shap_values, index=0, output_path=output_path)

        # Fairness
        dp = self.demographic_parity(y_pred, df[sensitive_column])
        eo = self.equal_opportunity(y, y_pred, df[sensitive_column])

        # Save fairness report
        with open(os.path.join(output_path, "fairness_report.txt"), "w") as f:
            f.write("=== Demographic Parity ===\n")
            f.write(str(dp) + "\n\n")

            f.write("=== Equal Opportunity ===\n")
            f.write(str(eo) + "\n")

        print("✅ All reports generated successfully!")


# =========================
# MAIN EXECUTION (OPTIONAL)
# =========================
# ...existing code...

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    
    from utils.config import MODEL_PATH, PROCESSED_DATA_PATH
    
    base_dir = Path(__file__).resolve().parent.parent  # project root

    # Use actual paths from config
    MODEL_PATH_ACTUAL = MODEL_PATH
    DATA_PATH = PROCESSED_DATA_PATH
    OUTPUT_PATH = base_dir / "outputs"

    # Verify model exists
    if not Path(MODEL_PATH_ACTUAL).exists():
        print(f"❌ Model not found at: {MODEL_PATH_ACTUAL}")
        print(f"   Run: python -m src.train_model")
        sys.exit(1)

    explainer = LoanModelExplainer(str(MODEL_PATH_ACTUAL))

    explainer.generate_reports(
        data_path=str(DATA_PATH),
        target_column="loan_status",      # CHANGED
        sensitive_column="addr_state",     # CHANGED
        output_path=str(OUTPUT_PATH),
    )