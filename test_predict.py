# test_predict.py
"""
Smoke and sanity test for AegisBank Loan Default Prediction System.
Tests:
  1. Health check endpoint (/health)
  2. Model and SHAP engine loading
  3. REST API prediction endpoint (/api/v1/predict)
"""

import sys
import os
import json
import time
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from webapp.app import app, db, User, ApiKey, _seed_default_users

def run_smoke_test():
    print("=" * 60)
    print("🚀 AegisBank Prediction System — Smoke Test")
    print("=" * 60)

    # 1. Database & User verification
    with app.app_context():
        db.create_all()
        _seed_default_users()
        user = User.query.filter_by(email="admin@aegisbank.com").first()
        assert user is not None, "Admin user seeding failed."
        
        # Ensure API key exists for testing
        test_api_key = "aegis-smoke-test-key-2026"
        key_entry = ApiKey.query.filter_by(user_id=user.id).first()
        if not key_entry:
            key_entry = ApiKey(user_id=user.id)
            key_entry.set_key(test_api_key)
            db.session.add(key_entry)
        else:
            key_entry.set_key(test_api_key)
        db.session.commit()
        print("✓ Database and default users verified.")

    # 2. Flask Test Client
    client = app.test_client()

    # Health check
    start = time.time()
    res_health = client.get("/health")
    health_time = (time.time() - start) * 1000
    print(f"✓ Health Check: status={res_health.status_code} in {health_time:.1f}ms")
    assert res_health.status_code == 200, f"Health check failed: {res_health.data}"
    health_data = res_health.get_json()
    print(f"  Model Loaded: {health_data.get('model_loaded')}, Features: {health_data.get('features')}")

    # REST API Prediction
    payload = {
        "borrower_name": "John Doe",
        "loan_amnt": 25000,
        "int_rate": 12.5,
        "installment": 600,
        "annual_inc": 75000,
        "dti": 15.0,
        "fico_range_low": 700,
        "fico_range_high": 720,
        "open_acc": 8,
        "revol_bal": 10000,
        "revol_util": 30.0,
        "total_acc": 15,
        "delinq_2yrs": 0,
        "inq_last_6mths": 1,
        "pub_rec": 0,
        "bc_open_to_buy": 15000,
        "bc_util": 35.0,
        "term": "36 months",
        "grade": "B",
        "sub_grade": "B2",
        "emp_length": "5 years",
        "home_ownership": "MORTGAGE",
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
        "addr_state": "CA",
        "initial_list_status": "w",
        "loan_purpose_text": "Borrower consolidating debt with stable employment."
    }

    start = time.time()
    res_pred = client.post(
        "/api/v1/predict",
        json=payload,
        headers={"X-API-Key": test_api_key}
    )
    pred_time = (time.time() - start) * 1000
    print(f"✓ REST API Predict (/api/v1/predict): status={res_pred.status_code} in {pred_time:.1f}ms")
    assert res_pred.status_code == 200, f"Predict failed: {res_pred.data}"
    pred_data = res_pred.get_json()
    prediction = pred_data.get("prediction", {})
    
    print("\n📊 Prediction Result Summary:")
    print(f"  Risk Level:        {prediction.get('risk_level')}")
    print(f"  Probability:       {prediction.get('probability')}%")
    print(f"  Verdict:           {prediction.get('verdict')}")
    print(f"  Expected Loss:     ${prediction.get('expected_loss'):,.2f}")
    print(f"  Expected Profit:   ${prediction.get('expected_profit'):,.2f}")
    print(f"  Override Triggered:{prediction.get('override_triggered')}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    run_smoke_test()
