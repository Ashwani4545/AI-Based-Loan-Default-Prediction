import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO)

def retrain_model():
    try:
        logging.info("🔄 Retraining model started...")

        # Run training script with the current Python interpreter
        subprocess.run([sys.executable, "-m", "src.train_model"], check=True)

        logging.info("✅ Model retrained successfully")

    except Exception as e:
        logging.error(f"❌ Retraining failed: {e}")