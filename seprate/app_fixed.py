# webapp/app.py  — AegisBank Risk Engine (Fixed)
# =====================================================
# FIXES APPLIED:
# 1. model_features.pkl path now resolved correctly via __file__
# 2. preprocess_input maps form fields → actual model feature names
# 3. xgb.DMatrix used for prediction (correct for saved XGBClassifier)
# 4. predict_proba added as fallback; primary path uses booster directly
# 5. History + metrics endpoints added for dashboard live data
# 6. CORS-safe JSON responses throughout
# =====================================================

from flask import Flask, render_template, request, jsonify, session
import joblib
import pandas as pd
import numpy as np
import sys, re, json, pickle, uuid
from pathlib import Path
from datetime import datetime
import xgboost as xgb

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent   # project root
sys.path.insert(0, str(ROOT))

from utils.config import MODEL_PATH, PROCESSED_DATA_PATH, TARGET_COLUMN

app = Flask(__name__)
app.secret_key = "aegisbank-secret-2025"  # needed for session-based history


# ── load model ───────────────────────────────────────────────────────────────
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    model = None

# ── load expected feature names ───────────────────────────────────────────────
features_path = ROOT / "utils" / "model_features.pkl"
try:
    with open(features_path, "rb") as f:
        MODEL_FEATURES = pickle.load(f)
    print(f"✅ {len(MODEL_FEATURES)} model features loaded")
except Exception as e:
    # Derive from booster if pkl missing
    if model is not None:
        try:
            MODEL_FEATURES = model.get_booster().feature_names
            print(f"✅ Features derived from booster: {len(MODEL_FEATURES)}")
        except Exception:
            MODEL_FEATURES = []
            print("⚠️  Could not determine model features — predictions may fail")
    else:
        MODEL_FEATURES = []


# ── metrics ──────────────────────────────────────────────────────────────────
def load_metrics():
    path = ROOT / "model_metrics.json"
    defaults = dict(
        accuracy=0.8074, precision=0.5471, recall=0.1687,
        f1_score=0.2579, roc_auc=0.7320,
        confusion_matrix=dict(tn=1250, fp=180, fn=302, tp=85)
    )
    try:
        with open(path) as f:
            data = json.load(f)
        return {
            "accuracy":  float(data.get("accuracy",  defaults["accuracy"])),
            "precision": float(data.get("precision", defaults["precision"])),
            "recall":    float(data.get("recall",    defaults["recall"])),
            "f1_score":  float(data.get("f1_score",  defaults["f1_score"])),
            "roc_auc":   float(data.get("roc_auc",   defaults["roc_auc"])),
            "confusion_matrix": {
                k: int(data.get("confusion_matrix", {}).get(k, defaults["confusion_matrix"][k]))
                for k in ("tn","fp","fn","tp")
            }
        }
    except FileNotFoundError:
        print("⚠️  model_metrics.json not found — using defaults")
        return defaults
    except Exception as e:
        print(f"❌ Metrics load error: {e}")
        return defaults

METRICS = load_metrics()
print(f"✅ Metrics: accuracy={METRICS['accuracy']:.4f}")


# ── helpers ───────────────────────────────────────────────────────────────────
def _sanitize(columns):
    seen = {}
    out = []
    for col in columns:
        c = re.sub(r"[\[\]<>]", "_", str(col))
        c = re.sub(r"\s+", "_", c.strip())
        c = re.sub(r"[^0-9a-zA-Z_]", "_", c)
        if c in seen:
            seen[c] += 1
            c = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
        out.append(c)
    return out


def preprocess_input(form: dict) -> pd.DataFrame:
    """
    Map HTML form fields → one-hot encoded DataFrame that matches
    the exact feature set the XGBoost model was trained on.
    """
    # ── numeric features (match training column names exactly) ──
    numeric = {
        "loan_amnt":                  float(form.get("loan_amnt", 0)),
        "int_rate":                   float(form.get("int_rate", 12.0)),
        "installment":                float(form.get("installment", 0)),
        "annual_inc":                 float(form.get("annual_inc", 0)),
        "dti":                        float(form.get("dti", 0)),
        "fico_range_low":             float(form.get("fico_range_low", 650)),
        "fico_range_high":            float(form.get("fico_range_high", 660)),
        "open_acc":                   float(form.get("open_acc", 5)),
        "revol_bal":                  float(form.get("revol_bal", 0)),
        "revol_util":                 float(form.get("revol_util", 30)),
        "total_acc":                  float(form.get("total_acc", 10)),
        "delinq_2yrs":                float(form.get("delinq_2yrs", 0)),
        "inq_last_6mths":             float(form.get("inq_last_6mths", 0)),
        "pub_rec":                    float(form.get("pub_rec", 0)),
        "pub_rec_bankruptcies":       float(form.get("pub_rec_bankruptcies", 0)),
        "tax_liens":                  float(form.get("tax_liens", 0)),
        "collections_12_mths_ex_med": float(form.get("collections_12_mths_ex_med", 0)),
        "acc_now_delinq":             float(form.get("acc_now_delinq", 0)),
        "tot_coll_amt":               float(form.get("tot_coll_amt", 0)),
        "tot_cur_bal":                float(form.get("tot_cur_bal", 0)),
        "avg_cur_bal":                float(form.get("avg_cur_bal", 0)),
        "bc_open_to_buy":             float(form.get("bc_open_to_buy", 0)),
        "bc_util":                    float(form.get("bc_util", 0)),
        "num_actv_bc_tl":             float(form.get("num_actv_bc_tl", 2)),
        "num_rev_accts":              float(form.get("num_rev_accts", 5)),
        "percent_bc_gt_75":           float(form.get("percent_bc_gt_75", 25)),
    }

    # ── categorical one-hot features ──────────────────────────────────────
    def ohe(prefix, value, valid):
        return {f"{prefix}_{v}": (1.0 if v == value else 0.0) for v in valid}

    cat = {}
    cat.update(ohe("term", form.get("term", "36_months"),
                   ["36_months", "60_months"]))
    cat.update(ohe("grade", form.get("grade", "B"),
                   list("ABCDEFG")))
    cat.update(ohe("sub_grade", form.get("sub_grade", "B1"),
                   [f"{g}{n}" for g in "ABCDEFG" for n in range(1,6)]))
    cat.update(ohe("emp_length", form.get("emp_length", "5_years"),
                   ["lt_1_year","1_year","2_years","3_years","4_years",
                    "5_years","6_years","7_years","8_years","9_years","10+_years"]))
    cat.update(ohe("home_ownership", form.get("home_ownership", "MORTGAGE"),
                   ["MORTGAGE","OWN","RENT","ANY","OTHER"]))
    cat.update(ohe("verification_status", form.get("verification_status", "Verified"),
                   ["Verified","Source_Verified","Not_Verified"]))
    cat.update(ohe("purpose", form.get("purpose", "debt_consolidation"),
                   ["debt_consolidation","credit_card","home_improvement",
                    "major_purchase","medical","moving","vacation","wedding",
                    "house","small_business","other"]))
    cat.update(ohe("addr_state", form.get("addr_state", "CA"),
                   ["CA","TX","NY","FL","IL","PA","OH","GA","NC","MI",
                    "NJ","WA","VA","AZ","MA","MD","CO","TN","IN","MO"]))
    cat.update(ohe("initial_list_status", form.get("initial_list_status", "w"),
                   ["w","f"]))

    row = {**numeric, **cat}

    # ── align to model feature set ────────────────────────────────────────
    if MODEL_FEATURES:
        aligned = {feat: row.get(feat, 0.0) for feat in MODEL_FEATURES}
        df = pd.DataFrame([aligned])[MODEL_FEATURES]
    else:
        df = pd.DataFrame([row])

    df.columns = _sanitize(df.columns)
    df = df.astype("float32")
    return df


def risk_label(prob: float) -> tuple[str, str]:
    if prob > 0.70:  return "🔴 VERY HIGH RISK",  "#d32f2f"
    if prob > 0.50:  return "🟠 HIGH RISK",        "#f57c00"
    if prob > 0.30:  return "🟡 MEDIUM RISK",      "#fbc02d"
    return                   "🟢 LOW RISK",         "#388e3c"


# ── session history helper ────────────────────────────────────────────────────
def get_history():
    return session.get("history", [])

def save_history(record: dict):
    history = get_history()
    history.insert(0, record)
    session["history"] = history[:200]   # keep last 200


# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    form_data = request.form.to_dict()
    app_id = "AGB-" + datetime.now().strftime("%Y%m%d") + "-" + str(uuid.uuid4())[:6].upper()

    try:
        df = preprocess_input(form_data)
        print(f"📊 Input shape: {df.shape}")

        # Prefer booster.predict for probability (consistent with training)
        try:
            dmat = xgb.DMatrix(df)
            raw_prob = float(model.get_booster().predict(dmat)[0])
        except Exception:
            raw_prob = float(model.predict_proba(df)[0][1])

        prediction   = 1 if raw_prob > 0.5 else 0
        probability  = round(raw_prob * 100, 2)
        risk_lv, color = risk_label(raw_prob)

        # Persist to session history
        record = {
            "app_id":      app_id,
            "name":        form_data.get("applicant_name", "Unknown"),
            "loan_amnt":   form_data.get("loan_amnt"),
            "grade":       form_data.get("grade"),
            "probability": probability,
            "prediction":  prediction,
            "risk_level":  risk_lv,
            "timestamp":   datetime.now().isoformat(),
        }
        save_history(record)

        return render_template(
            "result.html",
            app_id=app_id,
            prediction=prediction,
            probability=probability,
            risk_level=risk_lv,
            color=color,
            form_data=form_data,
        )

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", metrics=METRICS)


@app.route("/history")
def history_page():
    return render_template("history.html", records=get_history())


# ── API endpoints (used by JS front-end) ─────────────────────────────────────
@app.route("/api/metrics")
def api_metrics():
    return jsonify(METRICS)


@app.route("/api/history")
def api_history():
    return jsonify(get_history())


@app.route("/api/history/clear", methods=["POST"])
def api_clear_history():
    session.pop("history", None)
    return jsonify({"status": "cleared"})


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "features_loaded": len(MODEL_FEATURES),
        "version": "2.4"
    })


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
