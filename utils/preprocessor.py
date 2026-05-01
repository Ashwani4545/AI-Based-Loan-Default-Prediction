# utils/preprocessor.py
"""
Utility: extract and persist the feature list from the saved model.

Run once after training to regenerate utils/model_features.pkl.

Usage:
    python -m utils.preprocessor
"""

import sys
import pickle
import logging
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import MODEL_PATH, FEATURES_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def extract_and_save_features(model_path: str = MODEL_PATH, features_path: str = FEATURES_PATH) -> list:
    """Load model, extract feature names, save to pickle, return list."""
    model = joblib.load(model_path)

    try:
        # XGBoost
        feature_names = model.get_booster().feature_names
        log.info("Extracted %d features from XGBoost booster.", len(feature_names))
    except AttributeError:
        # sklearn estimators
        feature_names = list(getattr(model, "feature_names_in_", []))
        if not feature_names:
            raise RuntimeError("Cannot determine feature names from this model type.")
        log.info("Extracted %d features from sklearn estimator.", len(feature_names))

    Path(features_path).parent.mkdir(parents=True, exist_ok=True)
    with open(features_path, "wb") as f:
        pickle.dump(feature_names, f)
    log.info("Feature list saved → %s", features_path)

    return feature_names


if __name__ == "__main__":
    features = extract_and_save_features()
    print(f"✅  {len(features)} features saved.")