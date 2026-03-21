from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import MODEL_PATH, PROCESSED_DATA_PATH

app = Flask(__name__)

# Load model
model = joblib.load(MODEL_PATH)
print(f"✅ Model loaded from {MODEL_PATH}")

# Load training data to get feature names
training_data = pd.read_csv(PROCESSED_DATA_PATH)
expected_features = sorted([col for col in training_data.columns if col != 'loan_status'])
print(f"✅ Expected {len(expected_features)} features")

def preprocess_input(form_data):
    """Convert form input to one-hot encoded features matching training data"""
    
    # Initialize all features with 0
    input_dict = {feature: 0.0 for feature in expected_features}
    
    # Set numeric features
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
                input_dict[field] = float(form_data.get(field, 0))
            except:
                input_dict[field] = 0.0
    
    # One-hot encode categoricals
    categorical_mappings = {
        'term': form_data.get('term', '60_months'),
        'grade': form_data.get('grade', 'B'),
        'sub_grade': form_data.get('sub_grade', 'B1'),
        'emp_length': form_data.get('emp_length', '5_years'),
        'home_ownership': form_data.get('home_ownership', 'MORTGAGE'),
        'verification_status': form_data.get('verification_status', 'Verified'),
        'purpose': form_data.get('purpose', 'debt_consolidation'),
        'addr_state': form_data.get('addr_state', 'CA'),
        'initial_list_status': form_data.get('initial_list_status', 'w'),
        'earliest_cr_line': form_data.get('earliest_cr_line', '01-Jan-00'),
    }
    
    # Set one-hot encoded columns
    for cat_type, cat_value in categorical_mappings.items():
        if cat_type == 'term':
            col_name = f'term__{cat_value}'
        else:
            col_name = f'{cat_type}_{cat_value}'
        
        if col_name in input_dict:
            input_dict[col_name] = 1.0
    
    # Create DataFrame in correct order
    df = pd.DataFrame([input_dict])
    df = df[expected_features].astype('float32')
    
    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form.to_dict()
        print(f"📝 Form data received")
        
        # Preprocess to one-hot encoded features
        input_df = preprocess_input(form_data)
        print(f"📊 Input shape: {input_df.shape}, Features: {len(input_df.columns)}")
        
        # Make prediction
        dmatrix = xgb.DMatrix(input_df, enable_categorical=False)
        booster = model.get_booster()
        probability = booster.predict(dmatrix)[0]
        prediction = 1 if probability > 0.5 else 0
        
        print(f"✅ Prediction: {prediction}, Probability: {probability:.4f}")
        
        # Determine risk level
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
            prediction=int(prediction),
            probability=round(float(probability * 100), 2),
            risk_level=risk,
            color=color
        )
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000)