import sys
import json
import logging
from pathlib import Path
import pandas as pd
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import HISTORY_PATH, METRICS_PATH, PROCESSED_DATA_PATH
from monitoring.drift_detection import calculate_psi, FEATURE_COLUMNS, PSI_MEDIUM

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

RETRAIN_ACCURACY_THRESHOLD = 0.80
RETRAIN_DRIFT_THRESHOLD = PSI_MEDIUM

def monitor_health():
    log.info("🔍 Starting Model Health Check...")
    
    # 1. Load History
    try:
        with open(HISTORY_PATH) as f:
            history = json.load(f)
    except Exception:
        log.warning("No history found. Skipping health check.")
        return False

    if len(history) < 20:
        log.info("Not enough history (%d/20) for health check.", len(history))
        return False

    # 2. Performance Monitoring (using confirmed outcomes if available)
    df = pd.DataFrame(history)
    if "actual_outcome" in df.columns:
        valid_df = df.dropna(subset=["actual_outcome"])
        if len(valid_df) >= 10:
            accuracy = (valid_df["actual_outcome"] == valid_df["verdict"]).mean()
            log.info("Current Accuracy: %.2f%%", accuracy * 100)
            if accuracy < RETRAIN_ACCURACY_THRESHOLD:
                log.warning("🚨 Accuracy dropped below threshold (%.2f < %.2f)", accuracy, RETRAIN_ACCURACY_THRESHOLD)
                return trigger_retraining("low_accuracy")

    # 3. Drift Monitoring
    try:
        reference = pd.read_csv(PROCESSED_DATA_PATH)
        current = pd.DataFrame([entry["raw_input"] for entry in history[-100:]])
        
        high_drift_features = []
        for col in FEATURE_COLUMNS:
            if col in reference.columns and col in current.columns:
                psi = calculate_psi(reference[col].values, current[col].values)
                if psi >= RETRAIN_DRIFT_THRESHOLD:
                    high_drift_features.append(f"{col} (PSI={psi:.4f})")
        
        if high_drift_features:
            log.warning("🚨 High Drift detected in: %s", ", ".join(high_drift_features))
            return trigger_retraining("data_drift")
            
    except Exception as e:
        log.error("Error during drift monitoring: %s", e)

    log.info("✅ Model health is within acceptable limits.")
    return False

def trigger_retraining(reason):
    log.info("⚙️ Triggering Automated Retraining (Reason: %s)...", reason)
    retrain_script = Path(__file__).resolve().parent.parent / "webapp" / "retrain.py"
    try:
        # In a real environment, this might be a Celery task or a separate service
        # Here we just run it as a subprocess
        subprocess.Popen([sys.executable, str(retrain_script)], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL)
        log.info("✅ Retraining process started in background.")
        return True
    except Exception as e:
        log.error("Failed to start retraining: %s", e)
        return False

if __name__ == "__main__":
    monitor_health()
