from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import MODEL_PATH, PROCESSED_DATA_PATH

app = Flask(__name__)

# Load model from actual location
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.form.to_dict()

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Convert numeric columns
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return render_template(
            'result.html',
            prediction=int(prediction),
            probability=round(float(probability), 4)
        )

    except Exception as e:
        return render_template('error.html', error=str(e)), 400

# Dashboard page
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# Health check
@app.route('/health')
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)