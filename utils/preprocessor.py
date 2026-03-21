import pandas as pd
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import MODEL_PATH

# Load the model to extract feature names
import joblib
model = joblib.load(MODEL_PATH)
booster = model.get_booster()
model_features = booster.feature_names

print(f"Model expects {len(model_features)} features")

# Save the feature list
with open('utils/model_features.pkl', 'wb') as f:
    pickle.dump(model_features, f)
print("✅ Saved model features")