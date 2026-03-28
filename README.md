# AegisBank — Loan Default Prediction System

A production-grade ML web application for predicting loan default risk using XGBoost, with a modern Flask dashboard.

---

## Project Structure

```
aegisbank/
├── app.py                        # Flask application (all routes)
├── requirements.txt
├── model_metrics.json            # Auto-generated after training
│
├── data/
│   ├── raw/loan_data.csv         # Place your raw LendingClub CSV here
│   └── processed/cleaned_data.csv
│
├── models/
│   └── loan_default_model.pkl    # Saved best model
│
├── outputs/
│   ├── prediction_history.json   # Live prediction log
│   ├── drift_report.csv
│   └── drift_report.png
│
├── src/
│   ├── data_preprocessing.py     # Cleaning + feature engineering
│   ├── train_model.py            # Train LR + RF + XGBoost, save best
│   ├── evaluate_model.py         # Evaluate saved model → metrics JSON
│   ├── shap_explainer.py         # SHAP plots + fairness report
│   └── drift_detection.py        # PSI-based feature drift monitor
│
├── utils/
│   ├── config.py                 # All paths, params, constants
│   ├── preprocessor.py           # Extract & save feature list from model
│   └── model_features.pkl        # Auto-generated
│
├── templates/
│   ├── base.html                 # Shared nav + layout
│   ├── index.html                # Assessment form
│   ├── result.html               # Prediction result
│   ├── dashboard.html            # Model metrics dashboard
│   ├── history.html              # Prediction log with search/filter
│   ├── reports.html              # All borrower report cards
│   └── report_detail.html        # Individual printable report
│
└── static/
    ├── css/style.css
    └── js/script.js
```

---

## Setup & Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your dataset
Put your LendingClub-format CSV at:
```
data/raw/loan_data.csv
```
Required target column: `loan_status`

### 3. Preprocess data
```bash
python -m src.data_preprocessing
```

### 4. Train models
```bash
python -m src.train_model
```
This trains Logistic Regression, Random Forest, and XGBoost.  
Best model (by ROC-AUC) is saved to `models/loan_default_model.pkl`.  
Metrics are saved to `model_metrics.json`.

### 5. Save feature list (required for Flask app)
```bash
python -m utils.preprocessor
```

### 6. (Optional) Evaluate standalone
```bash
python -m src.evaluate_model
```

### 7. Run the web app
```bash
python app.py
```
Open http://127.0.0.1:5000

---

## Pages

| URL | Description |
|-----|-------------|
| `/` | Loan assessment form |
| `/predict` | POST — runs model, saves to history |
| `/dashboard` | Model metrics, confusion matrix, radar chart |
| `/history` | Filterable prediction log |
| `/reports` | Borrower report cards |
| `/reports/<id>` | Individual printable report |
| `/api/metrics` | JSON metrics |
| `/api/history` | JSON history (supports `?q=` search) |
| `/health` | Healthcheck |

---

## Data Flow

```
Raw CSV → data_preprocessing.py → cleaned_data.csv
         → train_model.py       → model.pkl + metrics.json + features.pkl
         → app.py               → /predict → history.json
                                          → result.html
```

The dashboard displays metrics from `model_metrics.json` — the exact same
values produced during training. No divergence between training and UI.

---

## Key Design Decisions

- **Model selection by ROC-AUC** (not accuracy) — better for imbalanced classes
- **Feature alignment** — model features extracted at training time and saved to `model_features.pkl`; inference always aligns to this list to prevent feature mismatch
- **Column sanitization** — XGBoost-safe column names (no `[]<>` chars) applied identically in training and inference
- **History** — JSON file (swap for SQLite/Postgres in production)
- **Print-ready reports** — `report_detail.html` has `@media print` styles
