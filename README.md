# AegisBank — AI-Based Loan Default Prediction System

> **v1 (current, implemented):** An XGBoost-powered ML web application that predicts probability of loan default in real time, with SHAP-based explainability, drift detection, and auto-retraining, trained on the LendingClub dataset.
>
> **v2 (this roadmap):** A rebuild around a **hybrid LSTM/RNN + XGBoost** architecture trained on real Indian bureau and cashflow data (CIBIL/Experian/CRIF/Equifax + Account Aggregator), moving from a static-snapshot model to a model that understands a borrower's *trajectory* over time — positioned as a sellable Risk Intelligence product for Indian NBFCs and digital lenders.

This README documents both: what exists today (v1) and the target architecture and phased plan to get to v2. Sections marked **[PLANNED]** are roadmap items, not yet implemented in code.

---

## 📌 Overview

Traditional credit scoring — and v1 of this project — treats a borrower as a **static snapshot**: one row of numbers (income, DTI, FICO, loan amount) at one point in time. That throws away the most informative signal in lending: **the shape of a borrower's financial behavior over time** — whether their bank balance has been sliding toward zero, whether their bureau repayment string shows a recent worsening pattern, whether credit utilization has been climbing month over month.

v2 keeps everything that works from v1 — XGBoost as the auditable, SHAP-explainable decision layer — and adds a **sequence model (LSTM/RNN)** that consumes real time-series data (monthly cashflow, bureau DPD history) and feeds its output into XGBoost as an additional signal. This is a **stacked ensemble**, not a black-box fusion: the LSTM extracts behavioral trend information, XGBoost makes the final, explainable decision.

- Uses **XGBoost** (Gradient Boosted Trees) as the final decision layer — auditable, SHAP-compatible
- Uses an **LSTM/RNN branch [PLANNED]** to learn from sequential bureau/cashflow history instead of hand-engineered summary stats
- Applies **SHAP (TreeSHAP)** for explainable, per-prediction reasoning on the final model
- Handles class imbalance using **SMOTE** and `scale_pos_weight`
- Detects **data drift** and triggers **retraining with human sign-off [PLANNED — v1 auto-retrains without review]**
- Is deployed through a **Flask** web interface for real-time predictions, with a path to a proper multi-tenant service **[PLANNED]**

---

## 🎯 Problem Statement

Financial institutions face key challenges in loan risk assessment:

- **Financial Exclusion** — Many individuals lack formal credit history (thin-file borrowers)
- **Sequence Blindness** — Static models miss deteriorating or improving trends in a borrower's behavior that only show up across time, not in a single snapshot
- **Hidden Bias** — AI models may introduce indirect discrimination
- **Accuracy vs Explainability Trade-off** — Accurate models are often black boxes
- **Model Drift** — Real-world prediction performance degrades over time
- **Regulatory Compliance** — AI systems must be transparent, auditable, and consent-driven when pulling bureau or bank data
- **Cost of Enterprise Tooling** — Existing enterprise-grade credit decisioning platforms (Experian, SAS, FIS) cost $500K–$5M+ with 6–18 month implementations, pricing out small NBFCs and digital lenders

**Objective:** Rebuild this into a fair, explainable, behaviorally-aware AI-based loan default prediction system, trained and validated on real Indian credit data, and priced for lenders the enterprise players don't serve.

---

## 🚀 Key Features

### Implemented (v1)

| Feature | Description |
|---|---|
| **AI Risk Prediction** | Predicts default probability using XGBoost |
| **Real-time WebSockets** | Streams prediction progress (Validating → Running → SHAP → Decision) |
| **Explainable AI (SHAP)** | Full SHAP integration showing exact risk drivers for each borrower |
| **Risk Classification** | LOW / MEDIUM / HIGH RISK bands with business override rules |
| **REST API + Swagger** | API Key authenticated `/api/v1/predict` with interactive docs at `/api/docs` |
| **Audit Log (flat file)** | Log of every decision, tracking the officer, inputs, and overrides — JSON-based, not yet immutable/production-grade |
| **Imbalance Handling** | SMOTE oversampling + XGBoost `scale_pos_weight` |
| **Drift Detection** | Monitors live prediction distribution vs reference data |
| **Auto-Retraining** | Triggers retraining on drift or every 100 new predictions — no human approval gate yet |
| **Feedback Loop** | Prediction history feeds back into training data |
| **Flask Web Interface** | Real-time predictions through an interactive web form |
| **Prediction History** | Searchable log of past assessments (JSON, capped at 500) |
| **Dashboard** | Model metrics, confusion matrix, and API Key management |
| **Borrower Reports** | Individual printable risk reports per prediction |

### Planned for v2

| Feature | Description |
|---|---|
| **LSTM/RNN Sequence Model** | Learns from monthly cashflow and bureau DPD history instead of static summary stats |
| **Stacked Ensemble (LSTM → XGBoost)** | LSTM output (probability or embedding) fed into XGBoost as an additional feature; XGBoost remains the final, explainable decision layer |
| **Real Bureau Data Integration** | Live pulls from CIBIL, Experian, CRIF High Mark, and Equifax via an aggregator API |
| **Account Aggregator (AA) Integration** | Consent-based real bank cashflow data via Setu, Finvu, or equivalent licensed AA, replacing synthetic income/DTI features |
| **Behavioral Trend Explanations** | Human-readable trend summaries (e.g. "balance declining over last 6 months") as a proxy explanation for the LSTM's contribution, alongside SHAP for the XGBoost layer |
| **Immutable Audit Trail** | Append-only, hashed decision log replacing the flat JSON file |
| **Champion/Challenger Retraining** | New model versions evaluated against a holdout set and require human approval before replacing production |
| **Real Database** | PostgreSQL replacing SQLite/JSON storage for applicants, sequences, predictions, and audit logs |
| **Fairness Auditing** | Outcome checks across income bands, gender, and geography proxies — demographic parity / equalized odds, not just a static report |
| **Multi-Tenant Support** | Per-lender data isolation and configurable risk thresholds |
| **Consent Management** | Captured and stored consent records for every bureau/AA data pull per regulatory retention requirements |

---

## 🧠 Machine Learning Pipeline

### v1 — Models Trained (implemented)

Three models are trained and compared on every training run:

| Model | Type |
|---|---|
| Logistic Regression | Linear baseline |
| Random Forest | Tree ensemble |
| **XGBoost** | **Gradient Boosted Trees — selected as best** |

> **Model selection:** The best model is chosen by **simulated profit score** (not just accuracy), because correctly catching a defaulter saves the full loan amount. XGBoost consistently wins in v1.

### v2 — Hybrid Architecture **[PLANNED]**

```
                 ┌─────────────────────────────┐
                 │   Sequence data (per        │
                 │   borrower, last 12 months) │
                 │   - Monthly bank balance     │
                 │   - Inflow / outflow         │
                 │   - EMI bounce flags         │
                 │   - Bureau DPD string        │
                 └──────────────┬───────────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │   LSTM / GRU branch  │
                     │   (1-2 layers,       │
                     │   heavy dropout)     │
                     └──────────┬───────────┘
                                │  output: probability
                                │  or embedding vector
                                ▼
┌───────────────────┐    ┌─────────────────────┐
│  Static features   │    │  LSTM output as an  │
│  (income, LTV,     │───▶│  extra XGBoost      │
│  bureau score,     │    │  input feature      │
│  employment, etc.) │    └──────────┬───────────┘
└───────────────────┘                │
                                      ▼
                          ┌───────────────────────┐
                          │   XGBoost (final,     │
                          │   SHAP-explainable    │
                          │   decision layer)      │
                          └──────────┬─────────────┘
                                     ▼
                        Probability of Default (PD)
                        + SHAP explanation
                        + behavioral trend summary
```

**Design rationale:**
- LSTM/GRU is a **feature extractor**, not the final decision-maker — this keeps XGBoost + SHAP as the auditable, regulator-facing explanation layer.
- A GRU is preferred over a full LSTM when historical data volume from a partner NBFC is limited (fewer parameters, less overfitting risk on small loan books).
- Borrowers with insufficient history (new-to-credit) get an explicit fallback path rather than a silently-imputed sequence.
- Model selection between XGBoost-alone, LSTM-alone, and the stacked ensemble is done on **precision**, not just recall — v1's 34.78% precision means two-thirds of flagged applicants are actually fine, which is a real business cost, not just a metric footnote.

### How Default is Predicted (v1, current thresholds — subject to re-tuning in v2)

The model outputs a **Probability of Default (PD)** between 0 and 1. Decision thresholds:

| Probability | Risk Label | Verdict |
|---|---|---|
| prob ≤ 0.40 | 🟢 LOW RISK | **Repay** — Loan likely to be repaid |
| 0.40 < prob ≤ 0.60 | 🟡 MEDIUM RISK | **Review** — Manual assessment recommended |
| prob > 0.60 | 🔴 HIGH RISK | **Default** — High probability of non-repayment |

**Business Override Rule:** If `Loan Amount > 5 × Annual Income`, the applicant is automatically flagged as **High Risk (Override)**, regardless of the model output.

> **[PLANNED]** Thresholds should become configurable per lender in v2 — different NBFCs have different risk appetites, and a single hardcoded cutoff won't sell across multiple customers.

### Features Used for Prediction

**Core Financial Features (v1, static):**
`loan_amnt`, `int_rate`, `installment`, `annual_inc`, `dti`, `fico_range_low`, `fico_range_high`, `revol_bal`, `revol_util`, `open_acc`, `total_acc`

**Credit Risk Indicators (v1, static):**
`delinq_2yrs`, `inq_last_6mths`, `pub_rec`, `pub_rec_bankruptcies`, `tax_liens`, `collections_12_mths_ex_med`, `acc_now_delinq`, `tot_coll_amt`, `tot_cur_bal`, `avg_cur_bal`, `bc_open_to_buy`, `bc_util`, `num_actv_bc_tl`, `num_rev_accts`, `percent_bc_gt_75`

**Categorical Features (one-hot encoded, v1):**
`term`, `grade`, `sub_grade`, `emp_length`, `home_ownership`, `verification_status`, `purpose`, `addr_state`, `initial_list_status`

**Engineered Features (v1, derived at both training and inference):**
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

**Sequence Features (v2, per month, last 12 months) [PLANNED]:**
| Feature | Source | Purpose |
|---|---|---|
| `monthly_balance` | Account Aggregator (bank statements) | Trend in liquidity |
| `monthly_inflow` / `monthly_outflow` | Account Aggregator | Cashflow stability |
| `emi_bounce_flag` | Account Aggregator | Missed-payment signal |
| `bureau_dpd_string` | CIBIL / Experian / CRIF / Equifax | Repayment trend, not just current score |
| `enquiry_count_trend` | Bureau | Credit-seeking acceleration |

**Target Column:** `loan_status` → `0` = Repay, `1` = Default

---

## 📊 Model Performance

> v1 metrics from `model_metrics.json`, trained on the LendingClub (US) dataset. **These numbers do not represent Indian borrower behavior and must be re-established once real Indian bureau/AA data is integrated (see Roadmap Phase 6).**

| Metric | Value |
|---|---|
| **Accuracy** | 68.68% |
| **Precision** | 34.78% |
| **Recall** | 64.77% |
| **F1-Score** | 45.25% |
| **ROC-AUC** | **74.06%** |
| **Best Model** | XGBoost |

### Confusion Matrix (v1)

```
                    Predicted: Repay   Predicted: Default
Actual: Repay          16,719 (TN)         7,284 (FP)
Actual: Default         2,113 (FN)         3,884 (TP)
```

> **Why high FP?** The dataset is imbalanced (~80% repay, ~20% default). The model is intentionally tuned to be conservative — in banking, it is safer to reject a good customer than to approve a defaulter. SMOTE and `scale_pos_weight` are used to improve recall on the minority (default) class. **Precision at 34.78% is the single biggest problem to solve in v2** — it means two-thirds of applicants flagged as risky are actually creditworthy, which is a direct business cost for any lender using this in production.

---

## 🔍 SHAP Explainability

**SHAP (SHapley Additive exPlanations)** is used to explain every prediction from the XGBoost layer.

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

**v2 note [PLANNED]:** SHAP stays on the XGBoost layer only — it is not applied to the LSTM branch directly (SHAP on recurrent nets is unreliable and hard to explain to a loan officer or regulator). Instead, the LSTM's contribution is surfaced as a plain-language behavioral trend summary (e.g. "bank balance declining over the last 6 months," "no missed payments in bureau history") shown alongside the SHAP feature list.

---

## 🏗️ System Architecture

### v1 — Current (implemented)
```
Browser (User)
    │
    │  GET /  →  index.html (loan assessment form)
    │
    │  SOCKET submit_prediction (Real-time Progress Stream)
    ▼
Flask app.py (SocketIO + Eventlet)
    ├── 1. Validate input (loan_amnt, annual_inc, fico)
    ├── 2. preprocess_input()  →  1-row DataFrame
    ├── 3. create_features_live()  →  engineered features
    ├── 4. add_economic_features()  →  macro context (currently hardcoded constants)
    ├── 5. reindex to MODEL_FEATURES  →  align columns
    ├── 6. SHAP explain_single()  →  exact feature drivers
    ├── 7. MODEL.predict_proba()[0][1]  →  PD probability
    ├── 8. Business Logic Overrides  →  apply Bank Policies
    ├── 9. Threshold logic  →  verdict + risk label
    ├── 10. Calculate LGD, EAD, Expected Loss
    ├── 11. Save to prediction_history.json
    ├── 12. log_decision()  →  audit log (logs/audit_log.json, flat file)
    ├── 13. SOCKET prediction_complete  →  redirect to results
    └── 14. render result.html  →  show risk + SHAP visualisations
```

### v2 — Target **[PLANNED]**
```
Browser / Partner NBFC System
    │
    │  Loan application submitted (with borrower consent)
    ▼
API Gateway (multi-tenant, per-lender auth)
    ├── 1. Consent capture & storage (bureau + AA pulls)
    ├── 2. Bureau connector  →  CIBIL / Experian / CRIF / Equifax (via aggregator API)
    ├── 3. AA connector      →  Setu / Finvu  →  real bank cashflow (consent-based)
    ├── 4. Build sequence tensor (12mo cashflow + bureau DPD history)
    ├── 5. Build static feature table (income, LTV, employment, bureau score)
    ├── 6. Sequence Scorer service (LSTM/GRU)  →  trend probability / embedding
    ├── 7. Final Scorer service (XGBoost)  →  PD probability + SHAP
    ├── 8. Threshold logic  →  per-tenant configurable verdict
    ├── 9. Write to PostgreSQL (applicants, sequences, predictions)
    ├── 10. Append-only, hashed audit log
    ├── 11. Champion/challenger check  →  human approval gate before any retrain ships
    └── 12. Return PD + SHAP + behavioral trend summary to lender dashboard
```

### 🔐 API Integration
The system exposes a secure REST API for B2B integrations:
- **Authentication**: `X-API-Key` or `Authorization: Bearer` header.
- **Endpoint**: `POST /api/v1/predict` (Accepts JSON, returns probability + expected loss).
- **Interactive Docs**: OpenAPI 3.0 specification served at `/api/docs`.
- **[PLANNED]** Bureau and AA connector endpoints, tenant-scoped API keys, and consent-record endpoints.

---

## 📁 Project Structure

```
AI-Based-Loan-Default-Prediction-main/
│
├── data/
│   ├── raw/
│   │   └── loan_dataset.csv          ← LendingClub CSV (v1) — replaced by real NBFC data in v2
│   └── processed/
│       ├── cleaned_data.csv          ← Auto-generated after preprocessing
│       └── sequences/                ← [PLANNED v2] per-borrower sequence tensors
│
├── connectors/                       ← [PLANNED v2]
│   ├── bureau_client.py              ← CIBIL / Experian / CRIF / Equifax via aggregator API
│   ├── aa_client.py                  ← Account Aggregator (Setu / Finvu) integration
│   └── consent_manager.py            ← Consent capture, storage, retention
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml
│
├── models/
│   ├── loan_default_model.pkl        ← Saved XGBoost model (v1)
│   └── sequence_model.h5             ← [PLANNED v2] Saved LSTM/GRU model
│
├── outputs/
│   ├── prediction_history.json       ← v1: flat JSON log (replaced by PostgreSQL in v2)
│   └── fairness_report.txt           ← v1: static report (replaced by ongoing fairness audit in v2)
│
├── reports/
│   └── <uuid>.txt                    ← Individual borrower risk reports
│
├── src/
│   ├── data_preprocessing.py         ← Clean + engineer + save processed CSV
│   ├── train_model.py                ← Train LR + RF + XGBoost, save best model
│   ├── train_sequence_model.py       ← [PLANNED v2] Train LSTM/GRU on sequence data
│   ├── train_ensemble.py             ← [PLANNED v2] Combine LSTM output + static features → XGBoost
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
│   ├── app.py                        ← Main Flask application (WebSockets + REST API)
│   ├── retrain.py                    ← v1: auto-retrain — [PLANNED v2] gated by champion/challenger approval
│   ├── aegisbank.db                  ← v1: SQLite — [PLANNED v2] PostgreSQL, multi-tenant
│   ├── templates/ ...                ← (unchanged from v1 — see below)
│   └── static/ ...                   ← (unchanged from v1)
│
├── monitoring/
│   └── drift_detection.py            ← PSI-based feature drift monitor
│
├── explainability/                   ← SHAP output plots directory
├── logs/                             ← [PLANNED v2] append-only, hashed audit logs
├── notebooks/                        ← Exploratory notebooks
│
├── feedback_loop.py                  ← Converts prediction history → training data
├── governance.py                     ← Logs every decision for compliance
├── model_metrics.json                ← Auto-generated training metrics
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Usage (v1, current)

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

### 5. Train the models
```bash
python -m src.train_model
```

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

```
| `/` | GET | Loan assessment input form |
| `/audit` | GET | Compliance Log Viewer (immutable trail — [PLANNED v2]) |
| `/api/docs` | GET | Swagger UI (interactive REST API documentation) |
| `/dashboard` | GET | Model metrics, confusion matrix, API Key Generation |
| `/history` | GET | Filterable log of all past predictions |
| `/timeline` | GET | Borrower historical risk tracking |
| `/reports` | GET | All borrower report cards |
| `/api/v1/predict` | POST | REST API endpoint for B2B integrations |
| `/health` | GET | Healthcheck endpoint |
```

---

## 🗺️ Roadmap to v2 (Hybrid LSTM/RNN + XGBoost, Real Indian Data)

### Phase 0 — Data & partnership groundwork (Weeks 1–4)
- Get sandbox access to a bureau aggregator API (covering CIBIL, Experian, CRIF High Mark, Equifax) and an Account Aggregator sandbox (Setu or Finvu).
- Secure an anonymized historical loan book from a partner NBFC/MFI, including monthly repayment/DPD history and the actual default label — this is the sequence data the LSTM branch depends on.
- Pick one MVP vertical (personal loans, MSME/GST lending, or microfinance) rather than building for all three at once.

### Phase 1 — Data engineering pipeline (Weeks 3–7)
- Build ingestion connectors: bureau API → structured credit report + tradeline history; AA API → categorized monthly cashflow; NBFC CSV → cleaned loan-performance table.
- Define the sequence tensor format (e.g. 12 months × N features per borrower) before any modeling work starts.
- Build an explicit fallback path for borrowers with insufficient history (new-to-credit).

### Phase 2 — Model architecture & training (Weeks 6–12)
- Build and validate the LSTM/GRU branch standalone against a naive trend baseline.
- Build the stacked ensemble: LSTM output/embedding feeds into XGBoost alongside static features.
- Re-tune model selection to prioritize precision (v1's 34.78% precision is the benchmark to beat).
- Keep SHAP on the XGBoost layer; add plain-language trend summaries for the LSTM's contribution.

### Phase 3 — Backend rebuild (Weeks 10–16)
- Replace flat-file storage with PostgreSQL.
- Split inference into independent Sequence Scorer and Final Scorer services.
- Move audit logging to append-only, hashed records.
- Replace auto-retraining with a champion/challenger workflow requiring human approval.

### Phase 4 — Security & compliance hardening (Weeks 14–20)
- Encrypt data at rest and in transit; role-based access control for PII vs. risk scores.
- Build consent capture and storage for every bureau/AA pull, retained per regulatory minimums.
- Run a fairness audit across income bands, gender, and geography proxies.

### Phase 5 — Multi-tenancy & productization (Weeks 18–26)
- Tenant isolation for data and model thresholds.
- Loan officer dashboard: applicant view, SHAP + trend explanation, cashflow chart, override workflow.
- Configurable per-lender risk thresholds.

### Phase 6 — Pilot & validate (Weeks 24–30)
- Run in shadow mode alongside a partner NBFC's existing manual process.
- Re-establish real precision/recall/profit-lift numbers on Indian data — replacing the LendingClub-based metrics above.

### Phase 7 — Go to market
- Price against the gap left by enterprise platforms ($500K–$5M+, 6–18 month implementations) — positioning as fast to deploy and built natively for Indian bureau + AA data.

---

## 🛠️ Technology Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.9+ |
| **ML Framework** | XGBoost, Scikit-learn |
| **Sequence Modeling [PLANNED]** | TensorFlow/Keras or PyTorch (LSTM/GRU) |
| **Explainability** | SHAP (TreeSHAP) |
| **Imbalance Handling** | imbalanced-learn (SMOTE) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Web Framework** | Flask, Flask-SocketIO, Eventlet |
| **API Docs** | Flask-Swagger-UI (OpenAPI 3.0) |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Serialization** | Joblib, Pickle (v1) / model registry [PLANNED v2] |
| **Database [PLANNED]** | PostgreSQL (replacing SQLite/JSON) |
| **Bureau Data [PLANNED]** | CIBIL, Experian, CRIF High Mark, Equifax via aggregator API |
| **Cashflow Data [PLANNED]** | Account Aggregator (Setu, Finvu, or equivalent licensed AA) |
| **Containerization** | Docker, Docker Compose |
| **Version Control** | Git / GitHub |

---

## 📋 Key Design Decisions

- **Model selection by profit, then by precision** — a model that correctly rejects one defaulter saves more than one that classifies many borderline cases correctly, but v1's low precision (34.78%) is a real cost that v2 must fix.
- **LSTM as feature extractor, not final decision-maker [PLANNED]** — keeps XGBoost + SHAP as the auditable, regulator-facing layer.
- **SMOTE + scale_pos_weight** — dual-layer imbalance handling for better recall on the minority (default) class.
- **Feature alignment** — `model_features.pkl` saves the exact ordered feature list at training time; inference `reindex()`s to this list.
- **Column sanitization** — XGBoost-safe column names applied identically in training and inference.
- **Governance logging** — every prediction logged with a trace ID, timestamp, inputs, and decision.
- **Drift detection** — compares live prediction distribution against training data using statistical tests.
- **Decision threshold = 0.40 (v1 default)** — conservative, since missing a defaulter (False Negative) is costlier than wrongly rejecting a good borrower (False Positive). Becomes per-tenant configurable in v2.

---

## ⚠️ Known Limitations (v1, being addressed by the v2 roadmap above)

- Trained on LendingClub (US) data — does not reflect Indian borrower behavior, income patterns, or loan products.
- Economic features (`inflation_rate`, `unemployment_rate`) are hardcoded constants — no real signal.
- Alternative credit features (`mobile_usage_score`, `digital_txn_count`, `utility_payment_score`) are filled with `0` at inference since they aren't actually collected.
- Prediction history is a flat JSON file — not suitable for production throughput or compliance-grade immutability.
- SHAP explanation is computed and stored but not yet displayed on the result page.
- Precision (34.78%) is low — roughly two-thirds of applicants flagged as risky are not actually defaulters.
- Auto-retraining has no human approval gate — risk of silently degrading the production model.
- No real bureau or bank-statement data integration — all financial inputs are user-entered or synthetic.
- Fairness reporting is a static, point-in-time report rather than an enforced or continuously monitored constraint.

---

## 📄 License

This project is developed for academic and research purposes.

---

## 👨‍💻 Made By

**Ashwani Pandey and team**
