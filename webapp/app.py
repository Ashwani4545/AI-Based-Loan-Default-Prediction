from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import sys
from pathlib import Path
import pickle
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import MODEL_PATH

app = Flask(__name__)

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load(MODEL_PATH)
print(f"✅ Model loaded from {MODEL_PATH}")

# Load expected features
with open('utils/model_features.pkl', 'rb') as f:
    model_features = pickle.load(f)
print(f"✅ Model expects {len(model_features)} features")


# ==============================
# LOAD METRICS (SINGLE SOURCE)
# ==============================
def load_metrics():
    try:
        metrics_path = Path(__file__).resolve().parent.parent / "model_metrics.json"

        with open(metrics_path, "r") as f:
            data = json.load(f)

        return data

    except Exception as e:
        print(f"❌ Error loading metrics: {e}")
        return {
            "accuracy": "N/A",
            "precision": "N/A",
            "recall": "N/A",
            "f1_score": "N/A",
            "confusion_matrix": {
                "tn": "N/A",
                "fp": "N/A",
                "fn": "N/A",
                "tp": "N/A"
            }
        }


# ==============================
# PREPROCESS INPUT
# ==============================
def preprocess_input(form_data):
    input_dict = {feature: 0.0 for feature in model_features}

    numeric_fields = {
        'loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
        'fico_range_low', 'fico_range_high', 'open_acc', 'revol_bal',
        'revol_util', 'total_acc', 'delinq_2yrs', 'inq_last_6mths',
        'pub_rec', 'pub_rec_bankruptcies', 'tax_liens',
        'collections_12_mths_ex_med', 'acc_now_delinq', 'tot_coll_amt',
        'tot_cur_bal', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util',
        'num_actv_bc_tl', 'num_rev_accts', 'percent_bc_gt_75'
    }

    for field in numeric_fields:
        if field in input_dict:
            try:
                value = float(form_data.get(field, 0))
                if value < 0:
                    raise ValueError(f"{field} cannot be negative")
                input_dict[field] = value
            except:
                input_dict[field] = 0.0

    categorical_mappings = {
        'term': form_data.get('term'),
        'grade': form_data.get('grade'),
        'sub_grade': form_data.get('sub_grade'),
        'emp_length': form_data.get('emp_length'),
        'home_ownership': form_data.get('home_ownership'),
        'verification_status': form_data.get('verification_status'),
        'purpose': form_data.get('purpose'),
        'addr_state': form_data.get('addr_state'),
        'initial_list_status': form_data.get('initial_list_status'),
        'earliest_cr_line': form_data.get('earliest_cr_line'),
    }

    for cat_type, cat_value in categorical_mappings.items():
        if not cat_value:
            continue

        if cat_type == 'term':
            col_name = f'term__{cat_value}'
        else:
            col_name = f'{cat_type}_{cat_value}'

        if col_name in input_dict:
            input_dict[col_name] = 1.0

    df = pd.DataFrame([input_dict])
    df = df[model_features].astype('float32')

    return df


# ==============================
# ROUTES
# ==============================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form.to_dict()
        print("📝 Form data received")

        input_df = preprocess_input(form_data)
        print(f"📊 Input shape: {input_df.shape}")

        # ✅ FIXED prediction logic
        probability = model.predict_proba(input_df)[0][1]
        prediction = int(probability > 0.5)

        print(f"✅ Prediction: {prediction}, Probability: {probability:.4f}")

        # Risk categorization
        if probability > 0.7:
            risk = "🔴 VERY HIGH RISK"
            color = "#d32f2f"
        elif probability > 0.5:
            risk = "🟠 HIGH RISK"
            color = "#f57c00"
        elif probability > 0.3:
            risk = "🟡 MEDIUM RISK"
            color = "#fbc02d"
        else:
            risk = "🟢 LOW RISK"
            color = "#388e3c"

        return render_template(
            'result.html',
            prediction=prediction,
            probability=round(float(probability * 100), 2),
            risk_level=risk,
            color=color
        )

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"error": str(e)}), 400


@app.route('/dashboard')
def dashboard():
    metrics = load_metrics()
    return render_template('dashboard.html', metrics=metrics)


@app.route('/health')
def health():
    return jsonify({"status": "ok"})


# ==============================
# RUN APP
# ==============================
if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000)