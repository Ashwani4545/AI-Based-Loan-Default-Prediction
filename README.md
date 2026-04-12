# AegisBank — Loan Default Prediction System

## Overview

The AI Loan Default Prediction System is a machine learning–based web application that helps financial institutions assess the risk of loan applicants. It predicts the likelihood of loan default using borrower and loan data. 

Unlike traditional credit scoring systems that rely on limited static records, this system uses an XGBoost-based model to capture complex patterns and improve prediction accuracy. It also incorporates Explainable AI (XAI) to provide transparency in decision-making. 

The system is deployed through a Flask web interface, enabling real-time predictions and making it a complete intelligent credit risk assessment solution.

## Problem Statement

Financial institutions face several challenges in loan risk assessment:

-   Financial Exclusion -- Many individuals lack formal credit history.
-   Hidden Bias -- AI models may introduce indirect discrimination.
-   Accuracy vs Explainability Trade-off -- Highly accurate models are
    often difficult to interpret.
-   Model Drift -- Prediction performance can degrade over time.
-   Regulatory Compliance -- AI systems must be transparent and
    auditable.

The objective is to develop a fair, explainable, and stable AI-based
loan default prediction system.

------------------------------------------------------------------------

## Objectives

1.  Develop a machine learning model to predict loan default risk.
2.  Improve prediction accuracy using XGBoost.
3.  Handle class imbalance using SMOTE.
4.  Integrate Explainable AI using SHAP.
5.  Build a Flask-based web interface for real-time prediction.
6.  Monitor model stability and detect drift.

------------------------------------------------------------------------

## 🚀 Key Features
### AI-Based Risk Prediction
Predicts loan default probability using an XGBoost-based machine learning model.
### Explainable AI (XAI)
Uses SHAP to provide clear insights into factors influencing each prediction.
### Robust Data Processing
Handles missing values, encoding, and feature scaling for reliable model input.
### Imbalance Handling
Applies SMOTE to improve detection of rare default cases.
### Web-Based Application
Interactive Flask interface for real-time loan risk prediction.
### Model Monitoring
Detects data drift to ensure long-term model stability and performance.

------------------------------------------------------------------------

## System Architecture

The system follows a multi-layer architecture:

1.  Data Layer
    -   Data collection
    -   Data cleaning
2.  Feature Engineering Layer
    -   Feature creation
    -   Feature transformation
3.  Modeling Layer
    -   Model training using XGBoost
    -   Hyperparameter tuning
4.  Governance & Explainability Layer
    -   SHAP explanations
    -   Feature importance analysis
5.  Deployment Layer
    -   Flask application
    -   Prediction API

------------------------------------------------------------------------

## Technology Stack

Programming Language: - Python

Machine Learning Libraries: - Scikit-learn - XGBoost - SHAP -
Imbalanced-learn

Data Processing: - Pandas - NumPy - Matplotlib - Seaborn

Web Development: - PyCharm- HTML - CSS - Bootstrap

Deployment: - GitHub - Render / Heroku / AWS

------------------------------------------------------------------------

## Project Structure
```
AI_Loan_Default_Predictor/
│
├── data/
│   ├── raw/
│   │   └── loan_data.csv
│   │
│   └── processed/
│       └── cleaned_data.csv
│
├── explainability/
│   └── shap_explainer.py
│
├── logs/
│
├── models/
│   └── loan_default_model.pkl
│
├── monitoring/
│   └── drift_detection.py
│
├── notebooks/
│   └── EDA.ipynb
│
├── outputs/
│
├── reports/
│
├── src/
│   ├── data_preprocessing.py
│   ├── generate_performance.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
│
├── utils/
│   ├── preprocessor.py
│   └── config.py
│
├── webapp/
│   ├── app.py                → Main Flask backend
│   ├── retrain.py
│   ├── model_training.ipynb          
│   ├── templates/            → HTML pages (UI)
│   │    ├── index.html        → Input form (user enters data)
│   │    ├── result.html       → Prediction result page
│   │    ├── base.html
│   │    ├── history.html      → Storing the history page
│   │    ├── reports.html      → Generating the reports page
│   │    ├── report_detail.html
│   │    └── dashboard.html    → Admin/analytics page
│   ├── static/               → Frontend assets
│       ├── css/style.css     → Styling
│       └── js/script.js      → JS logic (optional)
│
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── feedback_loop.py
├── governance.py
├── structure.txt
├── requirements.txt
└── README.md
```
------------------------------------------------------------------------

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

------------------------------------------------------------------------

## Running the Application

Start the Flask server

`python webapp/app.py`

Open your browser

`http://127.0.0.1:5000`

Enter borrower information to receive loan default prediction results.

------------------------------------------------------------------------

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

------------------------------------------------------------------------

## Data Flow

```
Raw CSV → data_preprocessing.py → cleaned_data.csv
         → train_model.py       → model.pkl + metrics.json + features.pkl
         → app.py               → /predict → history.json
                                          → result.html
```
The dashboard displays metrics from `model_metrics.json` — the exact same
values produced during training. No divergence between training and UI.

------------------------------------------------------------------------

## Key Design Decisions

- **Model selection by ROC-AUC** (not accuracy) — better for imbalanced classes
- **Feature alignment** — model features extracted at training time and saved to `model_features.pkl`; inference always aligns to this list to prevent feature mismatch
- **Column sanitization** — XGBoost-safe column names (no `[]<>` chars) applied identically in training and inference
- **History** — JSON file (swap for SQLite/Postgres in production)
- **Print-ready reports** — `report_detail.html` has `@media print` styles

------------------------------------------------------------------------

## Future Scope

The system can be expanded into a fintech platform offering:

-   Real-time credit scoring API
-   Integration with banking systems
-   Advanced monitoring dashboards
-   AI fairness auditing tools
-   Mobile application interface
-   MSME credit risk analytics

This could evolve into a Risk Intelligence SaaS platform for banks and
NBFCs.

------------------------------------------------------------------------

## Conclusion

The AI Loan Default Prediction System demonstrates how machine learning
can improve financial risk assessment.

By combining predictive modeling, explainable AI, and web deployment,
the system provides a transparent and intelligent framework for loan
decision-making.

This approach supports financial inclusion while maintaining fairness
and regulatory compliance.

------------------------------------------------------------------------

## License

This project is developed for academic and research purposes.
