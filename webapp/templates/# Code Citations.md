# Code Citations

## License: MIT

https://github.com/surya810/app-version-1.2/blob/2d60313bfc787c392f4de5c5b4f1bba7d2b00f20/templates/result.html

`````
The model expects the exact columns from training data (like `loan_amnt`, `int_rate`, etc.), not the form fields (`income`, `loan_amount`, `age`). Map form inputs to actual feature names and handle missing columns:

````python
# filepath: e:\AI-Based-Loan-Default-Prediction\webapp\app.py
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import sys
import re
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

def _sanitize_and_uniquify_columns(columns):
    seen = {}
    cleaned = []

    for col in columns:
        c = str(col)
        c = re.sub(r"[\[\]<>]", "_", c)
        c = re.sub(r"\s+", "_", c.strip())
        c = re.sub(r"[^0-9a-zA-Z_]", "_", c)

        if c in seen:
            seen[c] += 1
            c = f"{c}_{seen[c]}"
        else:
            seen[c] = 0

        cleaned.append(c)

    return cleaned

def preprocess_input(data):
    """Map form fields to model features"""
    # Create DataFrame with all numeric features
    df = pd.DataFrame({
        'loan_amnt': [float(data.get('loan_amount', 0))],
        'int_rate': [float(data.get('int_rate', 12.0))],  # Default 12%
        'installment': [float(data.get('existing_emi', 0))],
        'annual_inc': [float(data.get('income', 0))],
        'dti': [float(data.get('dti', 0.3))],  # Default 30%
        'fico_range_low': [float(data.get('credit_score', 650))],
        'fico_range_high': [float(data.get('credit_score', 650)) + 10],
        'open_acc': [float(data.get('open_acc', 5))],
        'revol_bal': [float(data.get('revol_bal', 0))],
        'revol_util': [float(data.get('revol_util', 30))],
        'total_acc': [float(data.get('total_acc', 10))],
        'delinq_2yrs': [float(data.get('delinq_2yrs', 0))],
        'inq_last_6mths': [float(data.get('inq_last_6mths', 0))],
        'pub_rec': [float(data.get('pub_rec', 0))],
        'pub_rec_bankruptcies': [float(data.get('pub_rec_bankruptcies', 0))],
        'tax_liens': [float(data.get('tax_liens', 0))],
        'collections_12_mths_ex_med': [float(data.get('collections_12_mths_ex_med', 0))],
        'acc_now_delinq': [float(data.get('acc_now_delinq', 0))],
        'tot_coll_amt': [float(data.get('tot_coll_amt', 0))],
        'tot_cur_bal': [float(data.get('tot_cur_bal', 0))],
        'avg_cur_bal': [float(data.get('avg_cur_bal', 0))],
        'bc_open_to_buy': [float(data.get('bc_open_to_buy', 0))],
        'bc_util': [float(data.get('bc_util', 0))],
        'num_actv_bc_tl': [float(data.get('num_actv_bc_tl', 2))],
        'num_rev_accts': [float(data.get('num_rev_accts', 5))],
        'percent_bc_gt_75': [float(data.get('percent_bc_gt_75', 25))],
    })

    # Categorical features (one-hot encoded)
    categorical_features = {
        f'home_ownership_{data.get("home_ownership", "MORTGAGE")}': [1],
        f'verification_status_{data.get("verification_status", "Verified")}': [1],
        f'purpose_{data.get("purpose", "personal")}': [1],
        f'emp_length_{data.get("emp_length", "5_years")}': [1],
    }

    for key, val in categorical_features.items():
        df[key] = val

    # Fill missing columns with 0
    df = df.astype("float32")

    return df

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

        # Preprocess input
        df = preprocess_input(data)

        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return render_template(
            'result.html',
            prediction=int(prediction),
            probability=round(float(probability), 4),
            risk_level="High" if probability > 0.6 else "Medium" if probability > 0.4 else "Low"
        )

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 400

# Dashboard page
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# Health check
@app.route('/health')
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5000)
`````

Then create [`webapp/templates/result.html`](webapp/templates/result.html):

`````html
<!doctype html>
<html>
  <head>
    <title>Prediction Result</title>
    <link
      rel="stylesheet"
      href="{{ url_for('static', filename='css/style.css') }}"
    />
  </head>
  <body>
    <div class="result-container">
      <h1>Assessment Result</h1>

      ``` ## License: MIT
      https://github.com/surya810/app-version-1.2/blob/2d60313bfc787c392f4de5c5b4f1bba7d2b00f20/templates/result.html
      ``` The model expects the exact columns from training data (like
      `loan_amnt`, `int_rate`, etc.), not the form fields (`income`,
      `loan_amount`, `age`). Map form inputs to actual feature names and handle
      missing columns: ````python # filepath:
      e:\AI-Based-Loan-Default-Prediction\webapp\app.py from flask import Flask,
      render_template, request, jsonify import joblib import pandas as pd import
      numpy as np import sys import re from pathlib import Path # Add project
      root to path sys.path.insert(0,
      str(Path(__file__).resolve().parent.parent)) from utils.config import
      MODEL_PATH, PROCESSED_DATA_PATH app = Flask(__name__) # Load model from
      actual location try: model = joblib.load(MODEL_PATH) print(f"✅ Model
      loaded from {MODEL_PATH}") except Exception as e: print(f"❌ Error loading
      model: {e}") model = None def _sanitize_and_uniquify_columns(columns):
      seen = {} cleaned = [] for col in columns: c = str(col) c =
      re.sub(r"[\[\]<>]", "_", c) c = re.sub(r"\s+", "_", c.strip()) c =
      re.sub(r"[^0-9a-zA-Z_]", "_", c) if c in seen: seen[c] += 1 c =
      f"{c}_{seen[c]}" else: seen[c] = 0 cleaned.append(c) return cleaned def
      preprocess_input(data): """Map form fields to model features""" # Create
      DataFrame with all numeric features df = pd.DataFrame({ 'loan_amnt':
      [float(data.get('loan_amount', 0))], 'int_rate':
      [float(data.get('int_rate', 12.0))], # Default 12% 'installment':
      [float(data.get('existing_emi', 0))], 'annual_inc':
      [float(data.get('income', 0))], 'dti': [float(data.get('dti', 0.3))], #
      Default 30% 'fico_range_low': [float(data.get('credit_score', 650))],
      'fico_range_high': [float(data.get('credit_score', 650)) + 10],
      'open_acc': [float(data.get('open_acc', 5))], 'revol_bal':
      [float(data.get('revol_bal', 0))], 'revol_util':
      [float(data.get('revol_util', 30))], 'total_acc':
      [float(data.get('total_acc', 10))], 'delinq_2yrs':
      [float(data.get('delinq_2yrs', 0))], 'inq_last_6mths':
      [float(data.get('inq_last_6mths', 0))], 'pub_rec':
      [float(data.get('pub_rec', 0))], 'pub_rec_bankruptcies':
      [float(data.get('pub_rec_bankruptcies', 0))], 'tax_liens':
      [float(data.get('tax_liens', 0))], 'collections_12_mths_ex_med':
      [float(data.get('collections_12_mths_ex_med', 0))], 'acc_now_delinq':
      [float(data.get('acc_now_delinq', 0))], 'tot_coll_amt':
      [float(data.get('tot_coll_amt', 0))], 'tot_cur_bal':
      [float(data.get('tot_cur_bal', 0))], 'avg_cur_bal':
      [float(data.get('avg_cur_bal', 0))], 'bc_open_to_buy':
      [float(data.get('bc_open_to_buy', 0))], 'bc_util':
      [float(data.get('bc_util', 0))], 'num_actv_bc_tl':
      [float(data.get('num_actv_bc_tl', 2))], 'num_rev_accts':
      [float(data.get('num_rev_accts', 5))], 'percent_bc_gt_75':
      [float(data.get('percent_bc_gt_75', 25))], }) # Categorical features
      (one-hot encoded) categorical_features = {
      f'home_ownership_{data.get("home_ownership", "MORTGAGE")}': [1],
      f'verification_status_{data.get("verification_status", "Verified")}': [1],
      f'purpose_{data.get("purpose", "personal")}': [1],
      f'emp_length_{data.get("emp_length", "5_years")}': [1], } for key, val in
      categorical_features.items(): df[key] = val # Fill missing columns with 0
      df = df.astype("float32") return df # Home page @app.route('/') def
      home(): return render_template('index.html') # Prediction route
      @app.route('/predict', methods=['POST']) def predict(): try: if model is
      None: return jsonify({"error": "Model not loaded"}), 500 data =
      request.form.to_dict() # Preprocess input df = preprocess_input(data) #
      Make prediction prediction = model.predict(df)[0] probability =
      model.predict_proba(df)[0][1] return render_template( 'result.html',
      prediction=int(prediction), probability=round(float(probability), 4),
      risk_level="High" if probability > 0.6 else "Medium" if probability > 0.4
      else "Low" ) except Exception as e: print(f"Prediction error: {e}") return
      jsonify({"error": str(e)}), 400 # Dashboard page @app.route('/dashboard')
      def dashboard(): return render_template('dashboard.html') # Health check
      @app.route('/health') def health(): return jsonify({"status": "ok",
      "model_loaded": model is not None}) if __name__ == '__main__':
      app.run(debug=False, host='127.0.0.1', port=5000)
    </div>
  </body>
</html>
`````

Then create [`webapp/templates/result.html`](webapp/templates/result.html):

`````html
<!doctype html>
<html>
  <head>
    <title>Prediction Result</title>
    <link
      rel="stylesheet"
      href="{{ url_for('static', filename='css/style.css') }}"
    />
  </head>
  <body>
    <div class="result-container">
      <h1>Assessment Result</h1>

      ``` ## License: MIT
      https://github.com/surya810/app-version-1.2/blob/2d60313bfc787c392f4de5c5b4f1bba7d2b00f20/templates/result.html
      ``` The model expects the exact columns from training data (like
      `loan_amnt`, `int_rate`, etc.), not the form fields (`income`,
      `loan_amount`, `age`). Map form inputs to actual feature names and handle
      missing columns: ````python # filepath:
      e:\AI-Based-Loan-Default-Prediction\webapp\app.py from flask import Flask,
      render_template, request, jsonify import joblib import pandas as pd import
      numpy as np import sys import re from pathlib import Path # Add project
      root to path sys.path.insert(0,
      str(Path(__file__).resolve().parent.parent)) from utils.config import
      MODEL_PATH, PROCESSED_DATA_PATH app = Flask(__name__) # Load model from
      actual location try: model = joblib.load(MODEL_PATH) print(f"✅ Model
      loaded from {MODEL_PATH}") except Exception as e: print(f"❌ Error loading
      model: {e}") model = None def _sanitize_and_uniquify_columns(columns):
      seen = {} cleaned = [] for col in columns: c = str(col) c =
      re.sub(r"[\[\]<>]", "_", c) c = re.sub(r"\s+", "_", c.strip()) c =
      re.sub(r"[^0-9a-zA-Z_]", "_", c) if c in seen: seen[c] += 1 c =
      f"{c}_{seen[c]}" else: seen[c] = 0 cleaned.append(c) return cleaned def
      preprocess_input(data): """Map form fields to model features""" # Create
      DataFrame with all numeric features df = pd.DataFrame({ 'loan_amnt':
      [float(data.get('loan_amount', 0))], 'int_rate':
      [float(data.get('int_rate', 12.0))], # Default 12% 'installment':
      [float(data.get('existing_emi', 0))], 'annual_inc':
      [float(data.get('income', 0))], 'dti': [float(data.get('dti', 0.3))], #
      Default 30% 'fico_range_low': [float(data.get('credit_score', 650))],
      'fico_range_high': [float(data.get('credit_score', 650)) + 10],
      'open_acc': [float(data.get('open_acc', 5))], 'revol_bal':
      [float(data.get('revol_bal', 0))], 'revol_util':
      [float(data.get('revol_util', 30))], 'total_acc':
      [float(data.get('total_acc', 10))], 'delinq_2yrs':
      [float(data.get('delinq_2yrs', 0))], 'inq_last_6mths':
      [float(data.get('inq_last_6mths', 0))], 'pub_rec':
      [float(data.get('pub_rec', 0))], 'pub_rec_bankruptcies':
      [float(data.get('pub_rec_bankruptcies', 0))], 'tax_liens':
      [float(data.get('tax_liens', 0))], 'collections_12_mths_ex_med':
      [float(data.get('collections_12_mths_ex_med', 0))], 'acc_now_delinq':
      [float(data.get('acc_now_delinq', 0))], 'tot_coll_amt':
      [float(data.get('tot_coll_amt', 0))], 'tot_cur_bal':
      [float(data.get('tot_cur_bal', 0))], 'avg_cur_bal':
      [float(data.get('avg_cur_bal', 0))], 'bc_open_to_buy':
      [float(data.get('bc_open_to_buy', 0))], 'bc_util':
      [float(data.get('bc_util', 0))], 'num_actv_bc_tl':
      [float(data.get('num_actv_bc_tl', 2))], 'num_rev_accts':
      [float(data.get('num_rev_accts', 5))], 'percent_bc_gt_75':
      [float(data.get('percent_bc_gt_75', 25))], }) # Categorical features
      (one-hot encoded) categorical_features = {
      f'home_ownership_{data.get("home_ownership", "MORTGAGE")}': [1],
      f'verification_status_{data.get("verification_status", "Verified")}': [1],
      f'purpose_{data.get("purpose", "personal")}': [1],
      f'emp_length_{data.get("emp_length", "5_years")}': [1], } for key, val in
      categorical_features.items(): df[key] = val # Fill missing columns with 0
      df = df.astype("float32") return df # Home page @app.route('/') def
      home(): return render_template('index.html') # Prediction route
      @app.route('/predict', methods=['POST']) def predict(): try: if model is
      None: return jsonify({"error": "Model not loaded"}), 500 data =
      request.form.to_dict() # Preprocess input df = preprocess_input(data) #
      Make prediction prediction = model.predict(df)[0] probability =
      model.predict_proba(df)[0][1] return render_template( 'result.html',
      prediction=int(prediction), probability=round(float(probability), 4),
      risk_level="High" if probability > 0.6 else "Medium" if probability > 0.4
      else "Low" ) except Exception as e: print(f"Prediction error: {e}") return
      jsonify({"error": str(e)}), 400 # Dashboard page @app.route('/dashboard')
      def dashboard(): return render_template('dashboard.html') # Health check
      @app.route('/health') def health(): return jsonify({"status": "ok",
      "model_loaded": model is not None}) if __name__ == '__main__':
      app.run(debug=False, host='127.0.0.1', port=5000)
    </div>
  </body>
</html>
`````

Then create [`webapp/templates/result.html`](webapp/templates/result.html):

`````html
<!doctype html>
<html>
  <head>
    <title>Prediction Result</title>
    <link
      rel="stylesheet"
      href="{{ url_for('static', filename='css/style.css') }}"
    />
  </head>
  <body>
    <div class="result-container">
      <h1>Assessment Result</h1>

      ``` ## License: MIT
      https://github.com/surya810/app-version-1.2/blob/2d60313bfc787c392f4de5c5b4f1bba7d2b00f20/templates/result.html
      ``` The model expects the exact columns from training data (like
      `loan_amnt`, `int_rate`, etc.), not the form fields (`income`,
      `loan_amount`, `age`). Map form inputs to actual feature names and handle
      missing columns: ````python # filepath:
      e:\AI-Based-Loan-Default-Prediction\webapp\app.py from flask import Flask,
      render_template, request, jsonify import joblib import pandas as pd import
      numpy as np import sys import re from pathlib import Path # Add project
      root to path sys.path.insert(0,
      str(Path(__file__).resolve().parent.parent)) from utils.config import
      MODEL_PATH, PROCESSED_DATA_PATH app = Flask(__name__) # Load model from
      actual location try: model = joblib.load(MODEL_PATH) print(f"✅ Model
      loaded from {MODEL_PATH}") except Exception as e: print(f"❌ Error loading
      model: {e}") model = None def _sanitize_and_uniquify_columns(columns):
      seen = {} cleaned = [] for col in columns: c = str(col) c =
      re.sub(r"[\[\]<>]", "_", c) c = re.sub(r"\s+", "_", c.strip()) c =
      re.sub(r"[^0-9a-zA-Z_]", "_", c) if c in seen: seen[c] += 1 c =
      f"{c}_{seen[c]}" else: seen[c] = 0 cleaned.append(c) return cleaned def
      preprocess_input(data): """Map form fields to model features""" # Create
      DataFrame with all numeric features df = pd.DataFrame({ 'loan_amnt':
      [float(data.get('loan_amount', 0))], 'int_rate':
      [float(data.get('int_rate', 12.0))], # Default 12% 'installment':
      [float(data.get('existing_emi', 0))], 'annual_inc':
      [float(data.get('income', 0))], 'dti': [float(data.get('dti', 0.3))], #
      Default 30% 'fico_range_low': [float(data.get('credit_score', 650))],
      'fico_range_high': [float(data.get('credit_score', 650)) + 10],
      'open_acc': [float(data.get('open_acc', 5))], 'revol_bal':
      [float(data.get('revol_bal', 0))], 'revol_util':
      [float(data.get('revol_util', 30))], 'total_acc':
      [float(data.get('total_acc', 10))], 'delinq_2yrs':
      [float(data.get('delinq_2yrs', 0))], 'inq_last_6mths':
      [float(data.get('inq_last_6mths', 0))], 'pub_rec':
      [float(data.get('pub_rec', 0))], 'pub_rec_bankruptcies':
      [float(data.get('pub_rec_bankruptcies', 0))], 'tax_liens':
      [float(data.get('tax_liens', 0))], 'collections_12_mths_ex_med':
      [float(data.get('collections_12_mths_ex_med', 0))], 'acc_now_delinq':
      [float(data.get('acc_now_delinq', 0))], 'tot_coll_amt':
      [float(data.get('tot_coll_amt', 0))], 'tot_cur_bal':
      [float(data.get('tot_cur_bal', 0))], 'avg_cur_bal':
      [float(data.get('avg_cur_bal', 0))], 'bc_open_to_buy':
      [float(data.get('bc_open_to_buy', 0))], 'bc_util':
      [float(data.get('bc_util', 0))], 'num_actv_bc_tl':
      [float(data.get('num_actv_bc_tl', 2))], 'num_rev_accts':
      [float(data.get('num_rev_accts', 5))], 'percent_bc_gt_75':
      [float(data.get('percent_bc_gt_75', 25))], }) # Categorical features
      (one-hot encoded) categorical_features = {
      f'home_ownership_{data.get("home_ownership", "MORTGAGE")}': [1],
      f'verification_status_{data.get("verification_status", "Verified")}': [1],
      f'purpose_{data.get("purpose", "personal")}': [1],
      f'emp_length_{data.get("emp_length", "5_years")}': [1], } for key, val in
      categorical_features.items(): df[key] = val # Fill missing columns with 0
      df = df.astype("float32") return df # Home page @app.route('/') def
      home(): return render_template('index.html') # Prediction route
      @app.route('/predict', methods=['POST']) def predict(): try: if model is
      None: return jsonify({"error": "Model not loaded"}), 500 data =
      request.form.to_dict() # Preprocess input df = preprocess_input(data) #
      Make prediction prediction = model.predict(df)[0] probability =
      model.predict_proba(df)[0][1] return render_template( 'result.html',
      prediction=int(prediction), probability=round(float(probability), 4),
      risk_level="High" if probability > 0.6 else "Medium" if probability > 0.4
      else "Low" ) except Exception as e: print(f"Prediction error: {e}") return
      jsonify({"error": str(e)}), 400 # Dashboard page @app.route('/dashboard')
      def dashboard(): return render_template('dashboard.html') # Health check
      @app.route('/health') def health(): return jsonify({"status": "ok",
      "model_loaded": model is not None}) if __name__ == '__main__':
      app.run(debug=False, host='127.0.0.1', port=5000)
    </div>
  </body>
</html>
`````

Then create [`webapp/templates/result.html`](webapp/templates/result.html):

`````html
<!doctype html>
<html>
  <head>
    <title>Prediction Result</title>
    <link
      rel="stylesheet"
      href="{{ url_for('static', filename='css/style.css') }}"
    />
  </head>
  <body>
    <div class="result-container">
      <h1>Assessment Result</h1>

      ``` ## License: MIT
      https://github.com/surya810/app-version-1.2/blob/2d60313bfc787c392f4de5c5b4f1bba7d2b00f20/templates/result.html
      ``` The model expects the exact columns from training data (like
      `loan_amnt`, `int_rate`, etc.), not the form fields (`income`,
      `loan_amount`, `age`). Map form inputs to actual feature names and handle
      missing columns: ````python # filepath:
      e:\AI-Based-Loan-Default-Prediction\webapp\app.py from flask import Flask,
      render_template, request, jsonify import joblib import pandas as pd import
      numpy as np import sys import re from pathlib import Path # Add project
      root to path sys.path.insert(0,
      str(Path(__file__).resolve().parent.parent)) from utils.config import
      MODEL_PATH, PROCESSED_DATA_PATH app = Flask(__name__) # Load model from
      actual location try: model = joblib.load(MODEL_PATH) print(f"✅ Model
      loaded from {MODEL_PATH}") except Exception as e: print(f"❌ Error loading
      model: {e}") model = None def _sanitize_and_uniquify_columns(columns):
      seen = {} cleaned = [] for col in columns: c = str(col) c =
      re.sub(r"[\[\]<>]", "_", c) c = re.sub(r"\s+", "_", c.strip()) c =
      re.sub(r"[^0-9a-zA-Z_]", "_", c) if c in seen: seen[c] += 1 c =
      f"{c}_{seen[c]}" else: seen[c] = 0 cleaned.append(c) return cleaned def
      preprocess_input(data): """Map form fields to model features""" # Create
      DataFrame with all numeric features df = pd.DataFrame({ 'loan_amnt':
      [float(data.get('loan_amount', 0))], 'int_rate':
      [float(data.get('int_rate', 12.0))], # Default 12% 'installment':
      [float(data.get('existing_emi', 0))], 'annual_inc':
      [float(data.get('income', 0))], 'dti': [float(data.get('dti', 0.3))], #
      Default 30% 'fico_range_low': [float(data.get('credit_score', 650))],
      'fico_range_high': [float(data.get('credit_score', 650)) + 10],
      'open_acc': [float(data.get('open_acc', 5))], 'revol_bal':
      [float(data.get('revol_bal', 0))], 'revol_util':
      [float(data.get('revol_util', 30))], 'total_acc':
      [float(data.get('total_acc', 10))], 'delinq_2yrs':
      [float(data.get('delinq_2yrs', 0))], 'inq_last_6mths':
      [float(data.get('inq_last_6mths', 0))], 'pub_rec':
      [float(data.get('pub_rec', 0))], 'pub_rec_bankruptcies':
      [float(data.get('pub_rec_bankruptcies', 0))], 'tax_liens':
      [float(data.get('tax_liens', 0))], 'collections_12_mths_ex_med':
      [float(data.get('collections_12_mths_ex_med', 0))], 'acc_now_delinq':
      [float(data.get('acc_now_delinq', 0))], 'tot_coll_amt':
      [float(data.get('tot_coll_amt', 0))], 'tot_cur_bal':
      [float(data.get('tot_cur_bal', 0))], 'avg_cur_bal':
      [float(data.get('avg_cur_bal', 0))], 'bc_open_to_buy':
      [float(data.get('bc_open_to_buy', 0))], 'bc_util':
      [float(data.get('bc_util', 0))], 'num_actv_bc_tl':
      [float(data.get('num_actv_bc_tl', 2))], 'num_rev_accts':
      [float(data.get('num_rev_accts', 5))], 'percent_bc_gt_75':
      [float(data.get('percent_bc_gt_75', 25))], }) # Categorical features
      (one-hot encoded) categorical_features = {
      f'home_ownership_{data.get("home_ownership", "MORTGAGE")}': [1],
      f'verification_status_{data.get("verification_status", "Verified")}': [1],
      f'purpose_{data.get("purpose", "personal")}': [1],
      f'emp_length_{data.get("emp_length", "5_years")}': [1], } for key, val in
      categorical_features.items(): df[key] = val # Fill missing columns with 0
      df = df.astype("float32") return df # Home page @app.route('/') def
      home(): return render_template('index.html') # Prediction route
      @app.route('/predict', methods=['POST']) def predict(): try: if model is
      None: return jsonify({"error": "Model not loaded"}), 500 data =
      request.form.to_dict() # Preprocess input df = preprocess_input(data) #
      Make prediction prediction = model.predict(df)[0] probability =
      model.predict_proba(df)[0][1] return render_template( 'result.html',
      prediction=int(prediction), probability=round(float(probability), 4),
      risk_level="High" if probability > 0.6 else "Medium" if probability > 0.4
      else "Low" ) except Exception as e: print(f"Prediction error: {e}") return
      jsonify({"error": str(e)}), 400 # Dashboard page @app.route('/dashboard')
      def dashboard(): return render_template('dashboard.html') # Health check
      @app.route('/health') def health(): return jsonify({"status": "ok",
      "model_loaded": model is not None}) if __name__ == '__main__':
      app.run(debug=False, host='127.0.0.1', port=5000)
    </div>
  </body>
</html>
`````

Then create [`webapp/templates/result.html`](webapp/templates/result.html):

`````html
<!doctype html>
<html>
  <head>
    <title>Prediction Result</title>
    <link
      rel="stylesheet"
      href="{{ url_for('static', filename='css/style.css') }}"
    />
  </head>
  <body>
    <div class="result-container">
      <h1>Assessment Result</h1>

      ``` ## License: MIT
      https://github.com/surya810/app-version-1.2/blob/2d60313bfc787c392f4de5c5b4f1bba7d2b00f20/templates/result.html
      ``` The model expects the exact columns from training data (like
      `loan_amnt`, `int_rate`, etc.), not the form fields (`income`,
      `loan_amount`, `age`). Map form inputs to actual feature names and handle
      missing columns: ````python # filepath:
      e:\AI-Based-Loan-Default-Prediction\webapp\app.py from flask import Flask,
      render_template, request, jsonify import joblib import pandas as pd import
      numpy as np import sys import re from pathlib import Path # Add project
      root to path sys.path.insert(0,
      str(Path(__file__).resolve().parent.parent)) from utils.config import
      MODEL_PATH, PROCESSED_DATA_PATH app = Flask(__name__) # Load model from
      actual location try: model = joblib.load(MODEL_PATH) print(f"✅ Model
      loaded from {MODEL_PATH}") except Exception as e: print(f"❌ Error loading
      model: {e}") model = None def _sanitize_and_uniquify_columns(columns):
      seen = {} cleaned = [] for col in columns: c = str(col) c =
      re.sub(r"[\[\]<>]", "_", c) c = re.sub(r"\s+", "_", c.strip()) c =
      re.sub(r"[^0-9a-zA-Z_]", "_", c) if c in seen: seen[c] += 1 c =
      f"{c}_{seen[c]}" else: seen[c] = 0 cleaned.append(c) return cleaned def
      preprocess_input(data): """Map form fields to model features""" # Create
      DataFrame with all numeric features df = pd.DataFrame({ 'loan_amnt':
      [float(data.get('loan_amount', 0))], 'int_rate':
      [float(data.get('int_rate', 12.0))], # Default 12% 'installment':
      [float(data.get('existing_emi', 0))], 'annual_inc':
      [float(data.get('income', 0))], 'dti': [float(data.get('dti', 0.3))], #
      Default 30% 'fico_range_low': [float(data.get('credit_score', 650))],
      'fico_range_high': [float(data.get('credit_score', 650)) + 10],
      'open_acc': [float(data.get('open_acc', 5))], 'revol_bal':
      [float(data.get('revol_bal', 0))], 'revol_util':
      [float(data.get('revol_util', 30))], 'total_acc':
      [float(data.get('total_acc', 10))], 'delinq_2yrs':
      [float(data.get('delinq_2yrs', 0))], 'inq_last_6mths':
      [float(data.get('inq_last_6mths', 0))], 'pub_rec':
      [float(data.get('pub_rec', 0))], 'pub_rec_bankruptcies':
      [float(data.get('pub_rec_bankruptcies', 0))], 'tax_liens':
      [float(data.get('tax_liens', 0))], 'collections_12_mths_ex_med':
      [float(data.get('collections_12_mths_ex_med', 0))], 'acc_now_delinq':
      [float(data.get('acc_now_delinq', 0))], 'tot_coll_amt':
      [float(data.get('tot_coll_amt', 0))], 'tot_cur_bal':
      [float(data.get('tot_cur_bal', 0))], 'avg_cur_bal':
      [float(data.get('avg_cur_bal', 0))], 'bc_open_to_buy':
      [float(data.get('bc_open_to_buy', 0))], 'bc_util':
      [float(data.get('bc_util', 0))], 'num_actv_bc_tl':
      [float(data.get('num_actv_bc_tl', 2))], 'num_rev_accts':
      [float(data.get('num_rev_accts', 5))], 'percent_bc_gt_75':
      [float(data.get('percent_bc_gt_75', 25))], }) # Categorical features
      (one-hot encoded) categorical_features = {
      f'home_ownership_{data.get("home_ownership", "MORTGAGE")}': [1],
      f'verification_status_{data.get("verification_status", "Verified")}': [1],
      f'purpose_{data.get("purpose", "personal")}': [1],
      f'emp_length_{data.get("emp_length", "5_years")}': [1], } for key, val in
      categorical_features.items(): df[key] = val # Fill missing columns with 0
      df = df.astype("float32") return df # Home page @app.route('/') def
      home(): return render_template('index.html') # Prediction route
      @app.route('/predict', methods=['POST']) def predict(): try: if model is
      None: return jsonify({"error": "Model not loaded"}), 500 data =
      request.form.to_dict() # Preprocess input df = preprocess_input(data) #
      Make prediction prediction = model.predict(df)[0] probability =
      model.predict_proba(df)[0][1] return render_template( 'result.html',
      prediction=int(prediction), probability=round(float(probability), 4),
      risk_level="High" if probability > 0.6 else "Medium" if probability > 0.4
      else "Low" ) except Exception as e: print(f"Prediction error: {e}") return
      jsonify({"error": str(e)}), 400 # Dashboard page @app.route('/dashboard')
      def dashboard(): return render_template('dashboard.html') # Health check
      @app.route('/health') def health(): return jsonify({"status": "ok",
      "model_loaded": model is not None}) if __name__ == '__main__':
      app.run(debug=False, host='127.0.0.1', port=5000)
    </div>
  </body>
</html>
`````

Then create [`webapp/templates/result.html`](webapp/templates/result.html):

`````html
<!doctype html>
<html>
  <head>
    <title>Prediction Result</title>
    <link
      rel="stylesheet"
      href="{{ url_for('static', filename='css/style.css') }}"
    />
  </head>
  <body>
    <div class="result-container">
      <h1>Assessment Result</h1>

      ``` ## License: MIT
      https://github.com/surya810/app-version-1.2/blob/2d60313bfc787c392f4de5c5b4f1bba7d2b00f20/templates/result.html
      ``` The model expects the exact columns from training data (like
      `loan_amnt`, `int_rate`, etc.), not the form fields (`income`,
      `loan_amount`, `age`). Map form inputs to actual feature names and handle
      missing columns: ````python # filepath:
      e:\AI-Based-Loan-Default-Prediction\webapp\app.py from flask import Flask,
      render_template, request, jsonify import joblib import pandas as pd import
      numpy as np import sys import re from pathlib import Path # Add project
      root to path sys.path.insert(0,
      str(Path(__file__).resolve().parent.parent)) from utils.config import
      MODEL_PATH, PROCESSED_DATA_PATH app = Flask(__name__) # Load model from
      actual location try: model = joblib.load(MODEL_PATH) print(f"✅ Model
      loaded from {MODEL_PATH}") except Exception as e: print(f"❌ Error loading
      model: {e}") model = None def _sanitize_and_uniquify_columns(columns):
      seen = {} cleaned = [] for col in columns: c = str(col) c =
      re.sub(r"[\[\]<>]", "_", c) c = re.sub(r"\s+", "_", c.strip()) c =
      re.sub(r"[^0-9a-zA-Z_]", "_", c) if c in seen: seen[c] += 1 c =
      f"{c}_{seen[c]}" else: seen[c] = 0 cleaned.append(c) return cleaned def
      preprocess_input(data): """Map form fields to model features""" # Create
      DataFrame with all numeric features df = pd.DataFrame({ 'loan_amnt':
      [float(data.get('loan_amount', 0))], 'int_rate':
      [float(data.get('int_rate', 12.0))], # Default 12% 'installment':
      [float(data.get('existing_emi', 0))], 'annual_inc':
      [float(data.get('income', 0))], 'dti': [float(data.get('dti', 0.3))], #
      Default 30% 'fico_range_low': [float(data.get('credit_score', 650))],
      'fico_range_high': [float(data.get('credit_score', 650)) + 10],
      'open_acc': [float(data.get('open_acc', 5))], 'revol_bal':
      [float(data.get('revol_bal', 0))], 'revol_util':
      [float(data.get('revol_util', 30))], 'total_acc':
      [float(data.get('total_acc', 10))], 'delinq_2yrs':
      [float(data.get('delinq_2yrs', 0))], 'inq_last_6mths':
      [float(data.get('inq_last_6mths', 0))], 'pub_rec':
      [float(data.get('pub_rec', 0))], 'pub_rec_bankruptcies':
      [float(data.get('pub_rec_bankruptcies', 0))], 'tax_liens':
      [float(data.get('tax_liens', 0))], 'collections_12_mths_ex_med':
      [float(data.get('collections_12_mths_ex_med', 0))], 'acc_now_delinq':
      [float(data.get('acc_now_delinq', 0))], 'tot_coll_amt':
      [float(data.get('tot_coll_amt', 0))], 'tot_cur_bal':
      [float(data.get('tot_cur_bal', 0))], 'avg_cur_bal':
      [float(data.get('avg_cur_bal', 0))], 'bc_open_to_buy':
      [float(data.get('bc_open_to_buy', 0))], 'bc_util':
      [float(data.get('bc_util', 0))], 'num_actv_bc_tl':
      [float(data.get('num_actv_bc_tl', 2))], 'num_rev_accts':
      [float(data.get('num_rev_accts', 5))], 'percent_bc_gt_75':
      [float(data.get('percent_bc_gt_75', 25))], }) # Categorical features
      (one-hot encoded) categorical_features = {
      f'home_ownership_{data.get("home_ownership", "MORTGAGE")}': [1],
      f'verification_status_{data.get("verification_status", "Verified")}': [1],
      f'purpose_{data.get("purpose", "personal")}': [1],
      f'emp_length_{data.get("emp_length", "5_years")}': [1], } for key, val in
      categorical_features.items(): df[key] = val # Fill missing columns with 0
      df = df.astype("float32") return df # Home page @app.route('/') def
      home(): return render_template('index.html') # Prediction route
      @app.route('/predict', methods=['POST']) def predict(): try: if model is
      None: return jsonify({"error": "Model not loaded"}), 500 data =
      request.form.to_dict() # Preprocess input df = preprocess_input(data) #
      Make prediction prediction = model.predict(df)[0] probability =
      model.predict_proba(df)[0][1] return render_template( 'result.html',
      prediction=int(prediction), probability=round(float(probability), 4),
      risk_level="High" if probability > 0.6 else "Medium" if probability > 0.4
      else "Low" ) except Exception as e: print(f"Prediction error: {e}") return
      jsonify({"error": str(e)}), 400 # Dashboard page @app.route('/dashboard')
      def dashboard(): return render_template('dashboard.html') # Health check
      @app.route('/health') def health(): return jsonify({"status": "ok",
      "model_loaded": model is not None}) if __name__ == '__main__':
      app.run(debug=False, host='127.0.0.1', port=5000)
    </div>
  </body>
</html>
`````

Then create [`webapp/templates/result.html`](webapp/templates/result.html):

````html
<!doctype html>
<html>
  <head>
    <title>Prediction Result</title>
    <link
      rel="stylesheet"
      href="{{ url_for('static', filename='css/style.css') }}"
    />
  </head>
  <body>
    <div class="result-container">
      <h1>Assessment Result</h1>

      ```
    </div>
  </body>
</html>
````
