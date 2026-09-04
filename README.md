# AegisBank — AI-Based Loan Default Prediction System

> **XGBoost · SHAP Explainability · Flask + SocketIO · Docker · Real-time WebSockets**

A full-stack AI web application that predicts the probability of loan default in real time. Built on the LendingClub dataset, it combines an XGBoost gradient-boosted model with SHAP-based explainability, live WebSocket prediction streaming, PSI drift detection, auto-retraining, a REST API, and a role-based multi-page Flask web interface.

---

## Table of Contents

- [Overview](#overview)
- [Live Features](#live-features)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Model Performance](#model-performance)
- [SHAP Explainability](#shap-explainability)
- [System Architecture](#system-architecture)
- [Web Application Pages](#web-application-pages)
- [REST API](#rest-api)
- [Role-Based Access Control](#role-based-access-control)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Docker](#docker)
- [Technology Stack](#technology-stack)
- [Design Decisions](#design-decisions)
- [License](#license)

---

## Overview

Most loan-risk tools treat a borrower as a static row of numbers. AegisBank does the same thing correctly — clean feature engineering, proper imbalance handling, SHAP-auditable decisions — and wraps it in a production-shaped application: WebSocket progress streaming, a REST API with API-key auth, PSI-based drift detection, auto-retraining, governance logging, and a multi-role web UI with dashboards, audit trails, batch upload, geographic heatmaps, and per-borrower printable reports.

**What the model actually does:**

1. Accepts a loan application (amounts, income, FICO, DTI, purpose, grade, etc.)
2. Engineers 8 derived features on top of 50+ raw LendingClub fields
3. Runs XGBoost `predict_proba` to get a Probability of Default (PD)
4. Runs TreeSHAP to identify the top-5 feature drivers
5. Applies a business-override rule (loan > 5× income → auto HIGH RISK)
6. Computes LGD, EAD, and Expected Loss
7. Stores the result and streams it back over WebSocket

---

## Live Features

| Feature | Detail |
|---|---|
| **AI Risk Prediction** | XGBoost trained on LendingClub data; decision threshold 0.3445 from `model_metrics.json` |
| **Real-time WebSockets** | Flask-SocketIO + Eventlet streams four progress stages: Validating → Running Model → Computing SHAP → Decision |
| **SHAP Explainability** | TreeSHAP top-5 feature impact per prediction; stored in history and audit log |
| **Risk Bands** | LOW (`prob ≤ 0.40`) · MEDIUM (`0.40–0.60`) · HIGH (`> 0.60`) |
| **Business Override** | Loan amount > 5× annual income → forced HIGH RISK regardless of model output |
| **Expected Loss** | PD × LGD × EAD computed per prediction |
| **REST API** | `POST /api/v1/predict` with `X-API-Key` / `Bearer` auth; OpenAPI 3.0 docs at `/api/docs` |
| **Role-Based Access** | Four roles: `analyst`, `risk_manager`, `admin`, `compliance` — enforced in nav and routes |
| **Drift Detection** | PSI-based monitor comparing live predictions vs. reference training distribution |
| **Auto-Retraining** | Triggers on drift or every 100 new predictions; reloads model into running server |
| **Feedback Loop** | `feedback_loop.py` converts confirmed-outcome predictions into new training rows |
| **Governance Logging** | `governance.py` appends every decision (trace ID, inputs, officer, overrides) to `logs/audit_log.json` |
| **Batch Upload** | CSV drag-and-drop; processes up to 50 rows; downloadable results CSV |
| **Prediction History** | Searchable, filterable table with actual-outcome confirmation buttons |
| **Borrower Reports** | Individual printable risk reports at `/reports/<id>` |
| **Borrower Timeline** | `/timeline` — chart + table of risk score history for a named borrower |
| **Geographic Heatmap** | `/heatmap` — state-level default probability map |
| **Borrower Comparison** | `/compare` — side-by-side risk profile for two borrowers |
| **Admin Panel** | User management (role promotion via SQLite), RBAC permissions matrix, system stats |
| **Docker** | `Dockerfile` + `docker-compose.yml` included |

---

## Machine Learning Pipeline

### Training flow

```
data/raw/loan_dataset.csv
        │
        ▼  src/data_preprocessing.py
data/processed/cleaned_data.csv
        │
        ▼  src/train_model.py
  Three models trained:
  ├── Logistic Regression   (linear baseline)
  ├── Random Forest         (tree ensemble)
  └── XGBoost               ← selected by simulated profit score
        │
        ▼
models/loan_default_model.pkl
utils/model_features.pkl
model_metrics.json
```

**Model selection criterion:** Simulated profit score, not raw accuracy. Catching a defaulter saves the full loan amount; the profit metric reflects that asymmetry directly.

### Engineered features

On top of 50+ raw LendingClub fields, eight features are derived at both training time (`data_preprocessing.py`) and inference time (`create_features_live()` in `app.py`):

| Feature | Formula | Purpose |
|---|---|---|
| `loan_to_income` | `loan_amnt / annual_inc` | Affordability ratio |
| `installment_to_income` | `installment / annual_inc` | Monthly payment burden |
| `credit_utilization` | `revol_bal / (revol_bal + bc_open_to_buy)` | Credit stress |
| `payment_capacity` | `annual_inc − (installment × 12)` | Free annual cash flow |
| `credit_stress` | `dti × loan_amnt` | Combined leverage |
| `high_dti_flag` | `1 if dti > 20` | Binary risk flag |
| `low_fico_flag` | `1 if fico < 600` | Binary credit flag |
| `recent_inquiries_flag` | `1 if inq_last_6mths > 3` | Credit-seeking signal |

Feature alignment at inference: `reindex(MODEL_FEATURES)` ensures column order matches exactly what the trained model saw.

### Imbalance handling

The LendingClub dataset is approximately 80% repay / 20% default. Two mechanisms address this:

- **SMOTE** — oversamples the minority (default) class in the training set
- **`scale_pos_weight`** — XGBoost parameter that further upweights the minority class

The decision threshold (0.3445 from `model_metrics.json`) is set conservatively: missing a defaulter (FN) is costlier than rejecting a good borrower (FP).

### Risk classification

| PD Range | Label | Verdict |
|---|---|---|
| `prob ≤ 0.40` | 🟢 LOW RISK | Repay — proceed with standard terms |
| `0.40 < prob ≤ 0.60` | 🟡 MEDIUM RISK | Review — manual assessment recommended |
| `prob > 0.60` | 🔴 HIGH RISK | Default — loan should be declined |

**Business override:** If `loan_amnt > 5 × annual_inc`, the verdict is forced to HIGH RISK and flagged as an override in the audit log.

---

## Model Performance

Metrics from `model_metrics.json` (XGBoost, LendingClub dataset):

| Metric | Value |
|---|---|
| Accuracy | 67.65% |
| Precision | 34.22% |
| Recall | 67.05% |
| F1-Score | 45.31% |
| **ROC-AUC** | **74.13%** |
| Decision threshold | 0.3445 |
| MSE | 0.2052 |
| RMSE | 0.4530 |
| MAE | 0.4131 |

### Confusion matrix (from `model_metrics.json`)

```
                      Predicted: Repay    Predicted: Default
Actual: Repay          16,274  (TN)          7,729  (FP)
Actual: Default         1,976  (FN)          4,021  (TP)
```

**Why the false-positive count is high:** The model is tuned for recall — catching actual defaulters — at the cost of precision. At 34.22% precision, roughly two out of three flagged applicants would actually have repaid. This is an explicit design choice for a lending risk tool where a missed default is more costly than a rejected good borrower. SMOTE and `scale_pos_weight` improve recall on the minority class.

---

## SHAP Explainability

Every prediction runs **TreeSHAP** (automatically selected for XGBoost) to compute per-feature attributions.

```
final_score   = Σ XGBoost tree outputs   (log-odds space)
probability   = sigmoid(final_score)
SHAP value_i  = feature i's contribution to (final_score − baseline_score)
```

- The top 5 features by absolute SHAP value are returned with each prediction
- Stored in `outputs/prediction_history.json` and `logs/audit_log.json`
- Visualised as a horizontal bar chart on the result page
- Positive SHAP → increases default probability; negative → decreases it

SHAP plots and a static fairness report are generated by `src/shap_explainer.py` and saved to the `explainability/` directory.

---

## System Architecture

### Prediction flow (WebSocket path)

```
Browser
  │  socket.emit('submit_prediction', formData)
  ▼
webapp/app.py  (Flask-SocketIO + Eventlet)
  ├─ 1.  validate_inputs()            ← field presence + numeric range
  ├─ 2.  preprocess_input()           ← build 1-row DataFrame
  ├─ 3.  create_features_live()       ← 8 engineered features
  ├─ 4.  add_economic_features()      ← macro context (constants, v1)
  ├─ 5.  df.reindex(MODEL_FEATURES)   ← align to training column order
  │       ↕ socket progress_update → "Running Model..."
  ├─ 6.  explain_single()             ← TreeSHAP top-5
  │       ↕ socket progress_update → "Computing SHAP..."
  ├─ 7.  model.predict_proba()[0][1]  ← PD probability
  ├─ 8.  business_override_rules()    ← loan_amnt > 5×income check
  ├─ 9.  threshold_logic()            ← risk label + verdict
  ├─ 10. compute_lgd_ead_el()         ← Expected Loss components
  ├─ 11. save to prediction_history.json
  ├─ 12. log_decision() → governance.py → audit_log.json
  └─ 13. socket.emit('prediction_complete') → redirect to /result
```

### Supporting processes

| Module | Role |
|---|---|
| `webapp/retrain.py` | Called on drift or every 100 predictions; re-runs `src.train_model`, reloads pkl into memory |
| `monitoring/drift_detection.py` | PSI comparison of live prediction distribution vs. reference data |
| `feedback_loop.py` | Reads `prediction_history.json`, converts confirmed outcomes into labelled training rows (capped at 500) |
| `governance.py` | Appends trace ID, timestamp, user, inputs, model output, overrides to `logs/audit_log.json` |
| `src/evaluate_model.py` | Evaluates saved pkl, overwrites `model_metrics.json` |
| `src/generate_performance_plots.py` | ROC curve and confusion matrix PNG output |
| `utils/config.py` | Central config: paths, thresholds, XGBoost hyperparameters |
| `utils/preprocessor.py` | Extracts and saves `utils/model_features.pkl` from trained model |

---

## Web Application Pages

| Route | Template | Description |
|---|---|---|
| `GET /` | `index.html` | Loan assessment form; submits over WebSocket with live progress modal |
| `GET /result` | `result.html` | Risk verdict, PD gauge, SHAP bar chart, LGD/EAD/EL cards, report link |
| `GET /dashboard` | `dashboard.html` | Model metrics tiles, confusion matrix, SHAP feature importance bars, drift status, API key generation — tabbed: Model Performance / MLOps |
| `GET /history` | `history.html` | Searchable, filterable prediction log; actual-outcome confirmation buttons |
| `GET /audit` | `audit.html` | Compliance audit trail (compliance and admin roles only) |
| `GET /reports` | `reports.html` | Card grid of all borrower reports |
| `GET /reports/<id>` | `report_detail.html` | Individual printable borrower risk report |
| `GET /compare` | `compare.html` | Side-by-side comparison of two borrowers |
| `GET /batch` | `batch.html` | CSV drag-and-drop batch scoring; downloadable results |
| `GET /heatmap` | `heatmap.html` | US state-level default probability heatmap |
| `GET /timeline` | `timeline.html` | Risk-score trend chart for a named borrower across multiple assessments |
| `GET /admin` | `admin.html` | User list, role promotion form, RBAC permissions matrix (admin only) |
| `GET /signin` | `signin.html` | Sign-in page (landing/marketing layout with particle canvas) |
| `GET /signup` | `signup.html` | Registration with email verification flow |
| `GET /forgot-password` | `forgot_password.html` | Password reset request |
| `GET /reset-password/<token>` | `reset_password.html` | Set new password with strength meter |
| `GET /api/docs` | — | Swagger UI (OpenAPI 3.0 spec from `webapp/swagger.json`) |
| `GET /health` | — | JSON health check endpoint |

---

## REST API

The system exposes a B2B prediction endpoint alongside the web UI.

```
POST /api/v1/predict
```

**Auth:** `X-API-Key: <key>` or `Authorization: Bearer <key>`
API keys are generated from the `/dashboard` page and stored in `webapp/aegisbank.db` (SQLite).

**Request body (JSON):**

```json
{
  "loan_amnt": 15000,
  "int_rate": 12.5,
  "annual_inc": 60000,
  "dti": 18.5,
  "fico_range_low": 700,
  "term": "36_months",
  "grade": "B",
  "purpose": "debt_consolidation"
}
```

**Response:**

```json
{
  "probability": 0.34,
  "risk_level": "LOW RISK",
  "verdict": "Repay",
  "expected_loss": 1240.50,
  "shap_top5": [
    {"feature": "dti", "value": 0.12},
    {"feature": "loan_to_income", "value": 0.09}
  ]
}
```

**Interactive docs:** `GET /api/docs` — full OpenAPI 3.0 spec with try-it-out.

---

## Role-Based Access Control

Four roles enforced in `base.html` navigation and route decorators:

| Feature | Analyst | Risk Manager | Admin | Compliance |
|---|---|---|---|---|
| Loan Assessment (`/`) | ✓ | ✓ | ✓ | — |
| Batch Upload | ✓ | ✓ | ✓ | — |
| Compare | ✓ | ✓ | ✓ | — |
| Dashboard | ✓ | ✓ | ✓ | ✓ |
| History | ✓ | ✓ | ✓ | ✓ |
| Reports | ✓ | ✓ | ✓ | ✓ |
| Heatmap / Timeline | ✓ | ✓ | ✓ | ✓ |
| Override Decisions | — | ✓ | ✓ | — |
| Audit Logs | — | — | ✓ | ✓ |
| Admin Panel | — | — | ✓ | — |
| User Management | — | — | ✓ | — |

User accounts and roles are stored in `webapp/aegisbank.db` (SQLite). Role changes persist across server restarts.

---

## Project Structure

```
AI-Based-Loan-Default-Prediction/
│
├── .github/
│   └── workflows/                 # CI/CD pipeline
│
├── data/
│   ├── raw/
│   │   └── loan_dataset.csv       # LendingClub-format input CSV
│   └── processed/
│       └── cleaned_data.csv       # Auto-generated by data_preprocessing.py
│
├── models/
│   └── loan_default_model.pkl     # Saved XGBoost model (auto-generated)
│
├── outputs/
│   ├── prediction_history.json    # Rolling prediction log (capped at 500)
│   └── fairness_report.txt        # Generated by shap_explainer.py
│
├── reports/
│   └── <uuid>.txt                 # Individual borrower risk report files
│
├── src/
│   ├── data_preprocessing.py      # Clean, engineer features, save cleaned_data.csv
│   ├── train_model.py             # Train LR + RF + XGBoost, select best, save pkl
│   ├── evaluate_model.py          # Evaluate saved model, write model_metrics.json
│   ├── shap_explainer.py          # SHAP summary plots + fairness_report.txt
│   └── generate_performance_plots.py  # ROC curve + confusion matrix PNGs
│
├── utils/
│   ├── config.py                  # Paths, thresholds, XGBoost hyperparameters
│   ├── preprocessor.py            # Extract + save model_features.pkl
│   └── model_features.pkl         # Ordered feature list (auto-generated)
│
├── webapp/
│   ├── app.py                     # Flask application: routes, SocketIO, API, auth
│   ├── retrain.py                 # Triggered auto-retrain logic
│   ├── swagger.json               # OpenAPI 3.0 spec
│   ├── aegisbank.db               # SQLite: users, roles, API keys
│   ├── templates/
│   │   ├── base.html              # Shared nav, session-aware, RBAC-driven links
│   │   ├── index.html             # Loan assessment form (WebSocket submit)
│   │   ├── result.html            # Risk verdict + SHAP visualisation
│   │   ├── dashboard.html         # Metrics, confusion matrix, drift, API keys
│   │   ├── history.html           # Searchable prediction log + outcome confirm
│   │   ├── audit.html             # Compliance audit trail
│   │   ├── reports.html           # Report card grid
│   │   ├── report_detail.html     # Individual printable report
│   │   ├── compare.html           # Side-by-side borrower comparison
│   │   ├── batch.html             # CSV batch upload + results
│   │   ├── heatmap.html           # Geographic risk heatmap
│   │   ├── timeline.html          # Borrower risk trend over time
│   │   ├── admin.html             # User management + RBAC matrix
│   │   ├── signin.html            # Sign-in (with marketing landing)
│   │   ├── signup.html            # Registration + email verification
│   │   ├── forgot_password.html   # Password reset request
│   │   └── reset_password.html    # Set new password + strength meter
│   └── static/
│       ├── css/style.css          # Full UI stylesheet (Syne + DM Sans + DM Mono)
│       └── js/script.js           # WebSocket handlers, gauge animation, counter FX
│
├── monitoring/
│   └── drift_detection.py         # PSI-based feature drift monitor
│
├── explainability/                # SHAP plot PNGs (auto-generated)
├── logs/
│   └── audit_log.json             # Append-only governance log
├── notebooks/                     # Exploratory analysis notebooks
│
├── feedback_loop.py               # Prediction history → labelled training rows
├── governance.py                  # Per-prediction compliance logging
├── model_metrics.json             # Auto-generated training metrics
├── challenger_metrics.json        # Challenger model comparison metrics
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── run.sh
├── test_predict.py                # Prediction unit tests
├── test_hf_integration.py         # Hugging Face Hub integration tests
└── update_html.py                 # Utility: bulk HTML template updates
```

---

## Setup & Usage

### Prerequisites

- Python 3.9+
- pip

### 1. Clone

```bash
git clone https://github.com/Ashwani4545/AI-Based-Loan-Default-Prediction.git
cd AI-Based-Loan-Default-Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies from `requirements.txt`:

```
flask>=3.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
joblib>=1.3.0
matplotlib>=3.7.0
imbalanced-learn>=0.11.0
shap>=0.43.0
huggingface_hub>=0.20.0
transformers>=4.30.0
```

Flask-SocketIO and Eventlet are also required for WebSocket support. Add them if not already present:

```bash
pip install flask-socketio eventlet flask-swagger-ui
```

### 3. Add dataset

Place a LendingClub-format CSV at:

```
data/raw/loan_dataset.csv
```

Required columns include: `loan_amnt`, `int_rate`, `installment`, `annual_inc`, `dti`, `fico_range_low`, `fico_range_high`, `revol_bal`, `revol_util`, `open_acc`, `total_acc`, `delinq_2yrs`, `inq_last_6mths`, `pub_rec`, `term`, `grade`, `sub_grade`, `emp_length`, `home_ownership`, `verification_status`, `purpose`, `addr_state`, `loan_status` (target: `0`=Repay, `1`=Default).

### 4. Preprocess

```bash
python -m src.data_preprocessing
```

Outputs `data/processed/cleaned_data.csv`.

### 5. Train

```bash
python -m src.train_model
```

Trains Logistic Regression, Random Forest, and XGBoost. Selects the best by simulated profit score. Saves `models/loan_default_model.pkl` and `model_metrics.json`.

### 6. Save feature list

```bash
python -m utils.preprocessor
```

Saves `utils/model_features.pkl` — the ordered column list used for inference alignment.

### 7. (Optional) SHAP plots + fairness report

```bash
python -m src.shap_explainer
```

Saves plots to `explainability/` and `outputs/fairness_report.txt`.

### 8. (Optional) Evaluate saved model

```bash
python -m src.evaluate_model
```

Overwrites `model_metrics.json` from the saved pkl.

### 9. Run the application

```bash
python webapp/app.py
```

Open: **http://127.0.0.1:5000**

Or use the included shell script:

```bash
bash run.sh
```

---

## Docker

```bash
docker-compose up --build
```

Open: **http://localhost:5000**

The `Dockerfile` uses Python 3.11-slim, installs requirements, copies the full project, and runs `webapp/app.py` via `python`.

---

## Technology Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.9+ |
| **ML — Core** | XGBoost 2.0+, scikit-learn 1.3+ |
| **ML — Explainability** | SHAP 0.43+ (TreeSHAP) |
| **ML — Imbalance** | imbalanced-learn 0.11+ (SMOTE) |
| **ML — NLP / Zero-shot** | Hugging Face `transformers` + `huggingface_hub` |
| **Data** | Pandas 2.0+, NumPy 1.24+ |
| **Visualisation** | Matplotlib 3.7+, Seaborn |
| **Web Framework** | Flask 3.0+, Flask-SocketIO, Eventlet |
| **API Docs** | Flask-Swagger-UI (OpenAPI 3.0) |
| **Frontend** | HTML5, CSS3 (custom design system), JavaScript (Socket.IO client, Chart.js) |
| **Storage** | SQLite (`aegisbank.db`), JSON flat files |
| **Serialisation** | Joblib, Pickle, Hugging Face Hub |
| **Containerisation** | Docker, Docker Compose |
| **CI/CD** | GitHub Actions (`.github/workflows/`) |

---

## Design Decisions

**Model selection by profit, not accuracy.** Accuracy on an imbalanced dataset is misleading. The training loop picks the model with the highest simulated profit score, which penalises missed defaults at their full loan value.

**Feature alignment via `reindex`.** `model_features.pkl` records the exact ordered column list at training time. Every inference call ends with `df.reindex(MODEL_FEATURES, fill_value=0)` to guarantee column order and handle any missing fields without silent errors.

**Decision threshold = 0.3445.** Lower than 0.5, deliberately. In a lending context, a False Negative (missed defaulter) carries a higher cost than a False Positive (rejected good borrower). The threshold is set at the value that maximised profit on the validation set.

**SMOTE + `scale_pos_weight`.** Two independent mechanisms for class imbalance — SMOTE at the data level, `scale_pos_weight` at the model level — giving the minority class sufficient signal without over-dependence on either technique alone.

**WebSocket progress streaming.** The prediction pipeline takes several seconds (preprocessing → SHAP → model). Rather than a blocking POST, the form submits over SocketIO and the server emits `progress_update` events at each stage, so the user sees a live progress modal.

**Flat JSON for storage.** `prediction_history.json` is capped at 500 entries. `audit_log.json` is append-only. Both are intentional for a single-machine deployment; a production multi-user deployment would replace these with a relational database.

**Governance logging decoupled from app.** `governance.py` is a standalone module called by `app.py`. This keeps compliance logging separate from business logic and easier to swap out for an immutable append-only store.

---

## License

This project is developed for academic and research purposes. See `LICENSE.txt` for full terms.

---

## Author

**Ashwani Pandey** — B.Tech Final Year Project
GitHub: [@Ashwani4545](https://github.com/Ashwani4545)