from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import MODEL_PATH, PROCESSED_DATA_PATH

app = Flask(__name__)

# Load model
model = joblib.load(MODEL_PATH)
print(f"✅ Model loaded from {MODEL_PATH}")

# Load training feature names
training_data = pd.read_csv(PROCESSED_DATA_PATH)
training_features = [col for col in training_data.columns if col != 'loan_status']
print(f"✅ Training features loaded: {len(training_features)} features")

def preprocess_input(data):
    """Convert form to match training preprocessing"""
    # Create DataFrame initialized to 0
    df = pd.DataFrame(0.0, index=[0], columns=training_features)
    
    # Numeric features
    numeric_map = {
        'loan_amnt': float(data.get('loan_amount', 5000)),
        'int_rate': float(data.get('int_rate', 12)),
        'installment': float(data.get('existing_emi', 200)),
        'annual_inc': float(data.get('income', 50000)),
        'dti': float(data.get('dti', 0.3)),
        'fico_range_low': float(data.get('credit_score', 700)),
        'fico_range_high': float(data.get('credit_score', 700)) + 10,
        'open_acc': float(data.get('open_acc', 5)),
        'revol_bal': float(data.get('revol_bal', 0)),
        'revol_util': float(data.get('revol_util', 30)),
        'total_acc': float(data.get('total_acc', 10)),
        'delinq_2yrs': float(data.get('delinq_2yrs', 0)),
        'inq_last_6mths': float(data.get('inq_last_6mths', 0)),
        'pub_rec': float(data.get('pub_rec', 0)),
        'pub_rec_bankruptcies': float(data.get('pub_rec_bankruptcies', 0)),
        'tax_liens': float(data.get('tax_liens', 0)),
        'collections_12_mths_ex_med': float(data.get('collections_12_mths_ex_med', 0)),
        'acc_now_delinq': float(data.get('acc_now_delinq', 0)),
        'tot_coll_amt': float(data.get('tot_coll_amt', 0)),
        'tot_cur_bal': float(data.get('tot_cur_bal', 0)),
        'avg_cur_bal': float(data.get('avg_cur_bal', 0)),
        'bc_open_to_buy': float(data.get('bc_open_to_buy', 0)),
        'bc_util': float(data.get('bc_util', 0)),
        'num_actv_bc_tl': float(data.get('num_actv_bc_tl', 2)),
        'num_rev_accts': float(data.get('num_rev_accts', 5)),
        'percent_bc_gt_75': float(data.get('percent_bc_gt_75', 25)),
    }
    
    for col, val in numeric_map.items():
        if col in df.columns:
            df[col] = val
    
    # One-hot encode categoricals
    term = data.get('term', '60_months')
    df[f'term__{term}'] = 1
    
    grade = data.get('grade', 'B')
    df[f'grade_{grade}'] = 1
    
    sub_grade = data.get('sub_grade', 'B1')
    df[f'sub_grade_{sub_grade}'] = 1
    
    emp_length = data.get('emp_length', '5_years')
    df[f'emp_length_{emp_length}'] = 1
    
    home_ownership = data.get('home_ownership', 'MORTGAGE')
    df[f'home_ownership_{home_ownership}'] = 1
    
    verification_status = data.get('verification_status', 'Verified')
    df[f'verification_status_{verification_status}'] = 1
    
    purpose = data.get('purpose', 'debt_consolidation')
    df[f'purpose_{purpose}'] = 1
    
    addr_state = data.get('addr_state', 'CA')
    df[f'addr_state_{addr_state}'] = 1
    
    initial_list_status = data.get('initial_list_status', 'w')
    df[f'initial_list_status_{initial_list_status}'] = 1
    
    # For earliest_cr_line - just set a random/default one
    df['earliest_cr_line_1_Jan'] = 1
    
    # Select only training features and ensure correct type
    df = df[training_features].astype('float32')
    
    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        df = preprocess_input(data)
        
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return render_template(
            'result.html',
            prediction=int(prediction),
            probability=round(float(probability * 100), 2),
            risk_level="🔴 HIGH" if probability > 0.6 else "🟡 MEDIUM" if probability > 0.4 else "🟢 LOW"
        )

    except Exception as e:
        print(f"Error: {e}")
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