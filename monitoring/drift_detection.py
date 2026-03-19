# drift_monitoring.py

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import PROCESSED_DATA_PATH

# ==============================
# 🔧 CONFIGURATION
# ==============================

PSI_LOW = 0.1       # No drift
PSI_MEDIUM = 0.25   # Moderate drift
NUM_BINS = 10

# These are actual columns from your processed data
FEATURE_COLUMNS = [
    'loan_amnt',
    'int_rate',
    'installment',
    'annual_inc',
    'dti',
    'fico_range_low',
    'open_acc',
    'revol_bal',
    'total_acc'
]

# ==============================
# 📥 DATA LOADER
# ==============================

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


# ==============================
# 📊 PSI CALCULATION
# ==============================

def calculate_psi(expected, actual, bins=10):

    def scale_range(arr):
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)

    expected = scale_range(expected)
    actual = scale_range(actual)

    breakpoints = np.linspace(0, 1, bins + 1)

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)

    # Avoid division errors
    expected_perc = np.where(expected_perc == 0, 0.0001, expected_perc)
    actual_perc = np.where(actual_perc == 0, 0.0001, actual_perc)

    psi_values = (expected_perc - actual_perc) * np.log(expected_perc / actual_perc)

    return np.sum(psi_values)


# ==============================
# 📌 PSI INTERPRETATION
# ==============================

def interpret_psi(psi_value):
    if psi_value < PSI_LOW:
        return "No Drift"
    elif psi_value < PSI_MEDIUM:
        return "Moderate Drift"
    else:
        return "High Drift"


# ==============================
# 🚨 DRIFT DETECTOR
# ==============================

class DriftDetector:

    def __init__(self, reference_df, new_df):
        self.reference_df = reference_df
        self.new_df = new_df

    def calculate_feature_drift(self):
        results = {}

        for col in FEATURE_COLUMNS:
            if col in self.reference_df.columns and col in self.new_df.columns:

                psi = calculate_psi(
                    self.reference_df[col].values,
                    self.new_df[col].values,
                    bins=NUM_BINS
                )

                results[col] = {
                    "psi_value": round(psi, 4),
                    "drift_status": interpret_psi(psi)
                }

        return results

    def overall_status(self, results):
        high_drift = any(
            res["drift_status"] == "High Drift"
            for res in results.values()
        )

        if high_drift:
            return "🚨 ALERT: Significant Drift Detected"
        else:
            return "✅ System Stable"


# ==============================
# ▶️ MAIN EXECUTION
# ==============================

def run_monitoring():

    print("\n🔍 Running Drift Monitoring...\n")

    # Load processed data
    data = load_data(str(PROCESSED_DATA_PATH))

    if data is None:
        print("❌ Failed to load data")
        return

    # Split into reference (70%) and new (30%)
    split_point = int(len(data) * 0.7)
    reference_data = data.iloc[:split_point]
    new_data = data.iloc[split_point:]

    detector = DriftDetector(reference_data, new_data)
    results = detector.calculate_feature_drift()

    print("📊 Feature Drift Report:\n")

    for feature, res in results.items():
        print(f"{feature}: PSI={res['psi_value']} → {res['drift_status']}")

    # Save to outputs folder
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "drift_report.csv"
    
    pd.DataFrame(results).T.to_csv(output_file)
    print(f"\n✅ Drift report saved to {output_file}")

    # Create visualization
    features = list(results.keys())
    psi_values = [results[f]["psi_value"] for f in features]
    drift_status = [results[f]["drift_status"] for f in features]

    # Color mapping
    colors = []
    for status in drift_status:
        if status == "No Drift":
            colors.append("green")
        elif status == "Moderate Drift":
            colors.append("orange")
        else:
            colors.append("red")

    # Create bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(features, psi_values, color=colors, alpha=0.7, edgecolor="black")
    plt.axhline(y=PSI_LOW, color="blue", linestyle="--", label=f"Low Drift Threshold ({PSI_LOW})")
    plt.axhline(y=PSI_MEDIUM, color="red", linestyle="--", label=f"Medium Drift Threshold ({PSI_MEDIUM})")
    plt.xlabel("Features", fontsize=12, fontweight="bold")
    plt.ylabel("PSI Value", fontsize=12, fontweight="bold")
    plt.title("Feature Drift Detection Report", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    # Save plot
    plot_file = output_dir / "drift_report.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"📊 Visualization saved to {plot_file}")
    plt.close()

    print("\n" + detector.overall_status(results))


# ==============================
# 🚀 RUN SCRIPT
# ==============================

if __name__ == "__main__":
    run_monitoring()