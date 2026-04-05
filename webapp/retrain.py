import subprocess
import logging

logging.basicConfig(level=logging.INFO)

def retrain_model():
    try:
        logging.info("🔄 Retraining model started...")

        # Run your existing training script
        subprocess.run(["python", "-m", "src.train_model"], check=True)

        logging.info("✅ Model retrained successfully")

    except Exception as e:
        logging.error(f"❌ Retraining failed: {e}")