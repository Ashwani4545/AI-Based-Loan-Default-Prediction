# AegisBank — AI-Based Loan Default Prediction System

> An XGBoost-powered machine learning web application that predicts the probability of loan default in real time, with SHAP-based explainability, drift detection, and auto-retraining.

---

## 📌 Overview

The **AegisBank Loan Default Prediction System** is a machine learning–based web application that helps financial institutions assess the risk of loan applicants. It predicts the **Probability of Default (PD)** for each borrower using financial and credit data.

Unlike traditional credit scoring systems that rely on static rules, this system:
- Uses **XGBoost** (Gradient Boosted Trees) to capture complex non-linear patterns
- Applies **SHAP (TreeSHAP)** for explainable, per-prediction reasoning
- Handles class imbalance using **SMOTE** and **scale_pos_weight**
- Detects **data drift** and triggers **automatic model retraining**
- Is deployed through a **Flask** web interface for real-time predictions

---

## 🎯 Problem Statement

Financial institutions face key challenges in loan risk assessment:

- **Financial Exclusion** — Many individuals lack formal credit history
- **Hidden Bias** — AI models may introduce indirect discrimination
- **Accuracy vs Explainability Trade-off** — Accurate models are often black boxes
- **Model Drift** — Real-world prediction performance degrades over time
- **Regulatory Compliance** — AI systems must be transparent and auditable

**Objective:** Develop a fair, explainable, and stable AI-based loan default prediction system.

---

## 🚀 Key Features

| Feature | Description |
|---|---|
| **AI Risk Prediction** | Predicts default probability using XGBoost |
| **Explainable AI (SHAP)** | Shows top 5 features driving each prediction |
| **Risk Classification** | LOW / MEDIUM / HIGH RISK bands with business override rules |
| **Imbalance Handling** | SMOTE oversampling + XGBoost `scale_pos_weight` |
| **Drift Detection** | Monitors live prediction distribution vs reference data |
| **Auto-Retraining** | Triggers model retraining on drift or every 100 new predictions |
| **Feedback Loop** | Prediction history feeds back into training data |
| **Governance Logging** | Every decision logged for audit and compliance |
| **Flask Web Interface** | Real-time predictions through an interactive web form |
| **Prediction History** | Full searchable log of all past assessments |
| **Dashboard** | Model metrics, confusion matrix visualised |
| **Borrower Reports** | Individual printable risk reports per prediction |

---

## 🧠 Machine Learning Pipeline

### Models Trained
Three models are trained and compared on every training run:

| Model | Type |
|---|---|
| Logistic Regression | Linear baseline |
| Random Forest | Tree ensemble |
| **XGBoost** | **Gradient Boosted Trees — selected as best** |

> **Model selection:** The best model is chosen by **simulated profit score** (not just accuracy), because correctly catching a defaulter saves the full loan amount. XGBoost consistently wins.

### How Default is Predicted

The model outputs a **Probability of Default (PD)** between 0 and 1. Decision thresholds:

| Probability | Risk Label | Verdict |
|---|---|---|
| prob ≤ 0.40 | 🟢 LOW RISK | **Repay** — Loan likely to be repaid |
| 0.40 < prob ≤ 0.60 | 🟡 MEDIUM RISK | **Review** — Manual assessment recommended |
| prob > 0.60 | 🔴 HIGH RISK | **Default** — High probability of non-repayment |

**Business Override Rule:** If `Loan Amount > 5 × Annual Income`, the applicant is automatically flagged as **High Risk (Override)**, regardless of the model output.

### Features Used for Prediction

**Core Financial Features:**
`loan_amnt`, `int_rate`, `installment`, `annual_inc`, `dti`, `fico_range_low`, `fico_range_high`, `revol_bal`, `revol_util`, `open_acc`, `total_acc`

**Credit Risk Indicators:**
`delinq_2yrs`, `inq_last_6mths`, `pub_rec`, `pub_rec_bankruptcies`, `tax_liens`, `collections_12_mths_ex_med`, `acc_now_delinq`, `tot_coll_amt`, `tot_cur_bal`, `avg_cur_bal`, `bc_open_to_buy`, `bc_util`, `num_actv_bc_tl`, `num_rev_accts`, `percent_bc_gt_75`

**Categorical Features (one-hot encoded):**
`term`, `grade`, `sub_grade`, `emp_length`, `home_ownership`, `verification_status`, `purpose`, `addr_state`, `initial_list_status`

**Engineered Features (derived at both training and inference):**
| Feature | Formula | Purpose |
|---|---|---|
| `loan_to_income` | `loan_amnt / annual_inc` | Affordability ratio |
| `installment_to_income` | `installment / annual_inc` | Monthly burden ratio |
| `credit_utilization` | `revol_bal / (revol_bal + bc_open_to_buy)` | Credit stress indicator |
| `payment_capacity` | `annual_inc - (installment × 12)` | Free cash flow |
| `credit_stress` | `dti × loan_amnt` | Combined leverage indicator |
| `high_dti_flag` | `1 if dti > 20 else 0` | Binary risk flag |
| `low_fico_flag` | `1 if fico < 600 else 0` | Binary credit risk flag |
| `recent_inquiries_flag` | `1 if inq_last_6mths > 3 else 0` | Credit-seeking behavior |

**Target Column:** `loan_status` → `0` = Repay, `1` = Default

---

## 📊 Model Performance

> Metrics from `model_metrics.json` — produced by the last training run.

| Metric | Value |
|---|---|
| **Accuracy** | 68.68% |
| **Precision** | 34.78% |
| **Recall** | 64.77% |
| **F1-Score** | 45.25% |
| **ROC-AUC** | **74.06%** |
| **Best Model** | XGBoost |

### Confusion Matrix

```
                    Predicted: Repay   Predicted: Default
Actual: Repay          16,719 (TN)         7,284 (FP)
Actual: Default         2,113 (FN)         3,884 (TP)
```

> **Why high FP?** The dataset is imbalanced (~80% repay, ~20% default). The model is intentionally tuned to be conservative — in banking, it is safer to reject a good customer than to approve a defaulter. SMOTE and `scale_pos_weight` are used to improve recall on the minority (default) class.

---

## 🔍 SHAP Explainability

**SHAP (SHapley Additive exPlanations)** is used to explain every prediction.

- Uses **TreeSHAP** — automatically selected for XGBoost models
- For each prediction, computes the **marginal contribution** of every feature to the model's output
- Returns the **top 5 features** with the highest absolute SHAP values
- These are stored in prediction history and governance logs

**How it works:**
```
final_score = sum of all XGBoost tree outputs (log-odds)
probability = sigmoid(final_score) = 1 / (1 + e^(-final_score))
SHAP value  = feature's share of (final_score - baseline_score)
```

Each SHAP value tells you: *"How much did this feature push the probability of default up or down compared to the average borrower?"*

---

## 🏗️ System Architecture

```
Browser (User)
    │
    │  GET /  →  index.html (loan assessment form)
    │
    │  POST /predict
    ▼
Flask app.py
    ├── 1. Validate input (loan_amnt, annual_inc, fico)
    ├── 2. preprocess_input()  →  1-row DataFrame
    ├── 3. create_features_live()  →  engineered features
    ├── 4. add_economic_features()  →  macro context
    ├── 5. reindex to MODEL_FEATURES  →  align columns
    ├── 6. SHAP explain_single()  →  top 5 feature drivers
    ├── 7. MODEL.predict_proba()[0][1]  →  PD probability
    ├── 8. Threshold logic  →  verdict + risk label
    ├── 9. Calculate LGD, EAD, Expected Loss
    ├── 10. Save to prediction_history.json
    ├── 11. log_decision()  →  governance audit log
    ├── 12. Feedback loop check  →  update training data if ≥100 entries
    ├── 13. Drift detection  →  compare vs reference data
    └── 14. render result.html  →  show risk to user
```

---

## 📁 Project Structure

```
AI-Based-Loan-Default-Prediction-main/
│
├── data/
│   ├── raw/
│   │   └── loan_dataset.csv          ← Place your LendingClub CSV here
│   └── processed/
│       └── cleaned_data.csv          ← Auto-generated after preprocessing
│
├── models/
│   └── loan_default_model.pkl        ← Saved best model (XGBoost)
│
├── outputs/
│   ├── prediction_history.json       ← Live log of all predictions
│   └── fairness_report.txt           ← Auto-generated fairness metrics
│
├── reports/
│   └── <uuid>.txt                    ← Individual borrower risk reports
│
├── src/
│   ├── data_preprocessing.py         ← Clean + engineer + save processed CSV
│   ├── train_model.py                ← Train LR + RF + XGBoost, save best model
│   ├── evaluate_model.py             ← Evaluate saved model, update metrics JSON
│   ├── shap_explainer.py             ← SHAP plots, fairness report, explainability
│   └── generate_performance_plots.py ← ROC curve, confusion matrix plots
│
├── utils/
│   ├── config.py                     ← All paths, thresholds, XGBoost params
│   ├── preprocessor.py               ← Extract & save feature list from model
│   └── model_features.pkl            ← Auto-generated list of model feature names
│
├── webapp/
│   ├── app.py                        ← Main Flask application (all routes)
│   ├── retrain.py                    ← Triggers retraining as subprocess
│   ├── model_training.ipynb          ← Jupyter notebook for exploratory training
│   ├── templates/
│   │   ├── signin.html               ← All user signin
│   │   ├── signup.html               ← All user signup
│   │   ├── welcome.html              ← Main page of our software
│   │   ├── signup.html               ← All user signup
│   │   ├── base.html                 ← Shared navigation + layout
│   │   ├── index.html                ← Loan assessment input form
│   │   ├── result.html               ← Prediction result page
│   │   ├── dashboard.html            ← Model metrics + confusion matrix
│   │   ├── history.html              ← Searchable prediction log
│   │   ├── reports.html              ← All borrower report cards
│   │   └── report_detail.html        ← Individual printable report
│   │
│   │
│   └── static/
│       ├── css/style.css             ← Application styling
│       └── js/script.js              ← Frontend JavaScript logic
│
├── monitoring/
│   └── drift_detection.py            ← PSI-based feature drift monitor
│
├── explainability/                   ← SHAP output plots directory
├── logs/                             ← Governance audit logs
├── notebooks/                        ← Exploratory notebooks
│
├── feedback_loop.py                  ← Converts prediction history → training data
├── governance.py                     ← Logs every decision for compliance
├── model_metrics.json                ← Auto-generated training metrics
├── Dockerfile                        ← Docker container definition
├── docker-compose.yml                ← Docker Compose configuration
├── requirements.txt                  ← Python dependencies
└── README.md
```

---

## ⚙️ Setup & Usage

### Prerequisites
- Python 3.9+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/your-username/AI-Based-Loan-Default-Prediction.git
cd AI-Based-Loan-Default-Prediction-main
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Place your dataset
Put your LendingClub-format CSV at:
```
data/raw/loan_dataset.csv
```
**Required target column:** `loan_status` (values: `0` = Repay, `1` = Default)

### 4. Preprocess the data
```bash
python -m src.data_preprocessing
```
This cleans the data, engineers features, and saves `data/processed/cleaned_data.csv`.

### 5. Train the models
```bash
python -m src.train_model
```
This trains Logistic Regression, Random Forest, and XGBoost.
Best model (by simulated profit) is saved to `models/loan_default_model.pkl`.
Metrics are saved to `model_metrics.json`.
Feature list is saved to `utils/model_features.pkl`.

### 6. Save the feature list (if not already generated)
```bash
python -m utils.preprocessor
```

### 7. (Optional) Generate SHAP plots and fairness report
```bash
python -m src.shap_explainer
```

### 8. (Optional) Evaluate the saved model
```bash
python -m src.evaluate_model
```

### 9. Run the web application
```bash
python webapp/app.py
```
Open your browser at: **http://127.0.0.1:5000**

---

## 🐳 Running with Docker

```bash
docker-compose up --build
```
Open: **http://localhost:5000**

---

## 🌐 Web Application Pages

| URL | Method | Description |
|---|---|---|
| `/` | GET | Loan assessment input form |
| `/predict` | POST | Runs model, saves to history, shows result |
| `/dashboard` | GET | Model metrics, confusion matrix, charts |
| `/history` | GET | Filterable log of all past predictions |
| `/reports` | GET | All borrower report cards |
| `/reports/<id>` | GET | Individual printable borrower report |
| `/api/metrics` | GET | JSON — model accuracy metrics |
| `/api/history` | GET | JSON history (supports `?q=` search filter) |
| `/health` | GET | Healthcheck endpoint |

---

## 🔄 Data Flow

```
data/raw/loan_dataset.csv
        ↓
src/data_preprocessing.py     → cleans, engineers features
        ↓
data/processed/cleaned_data.csv
        ↓
src/train_model.py             → trains 3 models, picks best
        ↓
models/loan_default_model.pkl  → saved XGBoost model
utils/model_features.pkl       → saved feature column list
model_metrics.json             → accuracy, AUC, confusion matrix
        ↓
webapp/app.py                  → Flask server loads model at startup
        ↓
User submits form → /predict route
        ↓
outputs/prediction_history.json → every prediction stored
        ↓
feedback_loop.py               → history becomes new training data
        ↓
monitoring/drift_detection.py  → detects distribution shift
        ↓
webapp/retrain.py              → triggers src.train_model if drift found
                                  or every 100 new predictions
```

---

## 🛠️ Technology Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.9+ |
| **ML Framework** | XGBoost, Scikit-learn |
| **Explainability** | SHAP (TreeSHAP) |
| **Imbalance Handling** | imbalanced-learn (SMOTE) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Web Framework** | Flask |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Serialization** | Joblib, Pickle |
| **Containerization** | Docker, Docker Compose |
| **Version Control** | Git / GitHub |

---

## 📋 Key Design Decisions

- **Model selection by profit** — not just ROC-AUC. A model that correctly rejects one defaulter saves more than one that classifies many borderline cases correctly.
- **SMOTE + scale_pos_weight** — dual-layer imbalance handling for better recall on the minority (default) class.
- **Feature alignment** — `model_features.pkl` saves the exact ordered feature list at training time. At inference, all inputs are `reindex()`-ed to this list to prevent column mismatch.
- **Column sanitization** — XGBoost-safe column names (removes `[]<>` characters and spaces) applied identically in both training and inference pipelines.
- **Governance logging** — every prediction is logged with a trace ID, timestamp, input features, and decision for compliance and auditing.
- **Drift detection** — compares live prediction distribution against the first 10,000 rows of training data using statistical tests.
- **History cap** — prediction history is capped at 500 entries in JSON. For production, replace with a database (SQLite / PostgreSQL).
- **Decision threshold = 0.40** — set conservatively; in banking, missing a defaulter (False Negative) is more costly than wrongly rejecting a good borrower (False Positive).

---

## 🔮 Future Scope

- Replace JSON history file with a proper database (PostgreSQL / SQLite)
- Add real-time macroeconomic indicators (live inflation, unemployment rate API)
- Expose a REST API for integration with banking core systems
- Add mobile-friendly responsive UI improvements
- Implement advanced fairness auditing (demographic parity, equalized odds)
- Add role-based access control (loan officer vs admin vs auditor)
- Extend to MSME credit risk analytics
- Evolve into a Risk Intelligence SaaS platform for banks and NBFCs

---

## ⚠️ Known Limitations

- Economic features (`inflation_rate`, `unemployment_rate`) are hardcoded constants — they carry no real signal since they are the same for every prediction.
- Alternative credit features (`mobile_usage_score`, `digital_txn_count`, `utility_payment_score`) are currently filled with `0` at inference since these fields are not collected from users.
- Prediction history is stored as a flat JSON file — not suitable for high-throughput production environments.
- The SHAP explanation is computed and stored in history but is not yet displayed on the result page.

---

Screenshots
---

## 📄 License

This project is developed for academic and research purposes.

---

## 👨‍💻 Project

**AegisBank — AI Loan Default Prediction**  
Built with Flask · XGBoost · SHAP · Python

---

## 👨‍💻 Made By

**Ashwani Pandey and team**  
