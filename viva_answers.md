# 🎓 VIVA Q&A — AI-Based Loan Default Prediction
### Project: AegisBank Loan Default Prediction System

---

## ❓ Q1: How is the model predicting that a person will DEFAULT? What parameters are used and why?

### How the Model Flags a Defaulter

The model looks at a person's financial profile and outputs a **probability score between 0 and 1** — called the **Probability of Default (PD)**. Based on that score, the system flags the applicant:

| Probability | Risk Label | Decision |
|---|---|---|
| prob ≤ 0.40 | LOW RISK | **Repay** → Not a defaulter |
| 0.40 < prob ≤ 0.60 | MEDIUM RISK | **Review** → Manual check needed |
| prob > 0.60 | HIGH RISK | **Default** → Flagged as defaulter |

Additionally, a **Business Override Rule** auto-flags someone as High Risk if:
```
Loan Amount > 5 × Annual Income  (regardless of what the model says)
```

---

### Parameters Used and WHY Each One Matters

| Parameter | Why It Predicts Default |
|---|---|
| `loan_amnt` | Bigger loans → harder to repay → more default risk |
| `int_rate` | High interest → larger repayment burden → default risk |
| `annual_inc` | Low income → can't afford repayments |
| `dti` (Debt-to-Income Ratio) | Already over-leveraged → can't take more debt |
| `fico_range_low` | Credit score = historical repayment behavior (most important!) |
| `installment` | Monthly burden — if too high vs income, likely to default |
| `revol_bal` | High revolving balance = already maxed on credit |
| `revol_util` | High credit utilization = financial stress |
| `delinq_2yrs` | Past missed payments = strong predictor of future default |
| `pub_rec` | Bankruptcies / civil judgments = very high risk signal |
| `inq_last_6mths` | Many recent credit inquiries = desperate for credit = risky |
| `open_acc` / `total_acc` | Too many open accounts = overextended |
| `grade` / `sub_grade` | LendingClub's own internal risk grading |
| `purpose` | "debt_consolidation" applicants historically default more often |
| `home_ownership` | OWN vs RENT — asset backing reduces default risk |
| `emp_length` | Longer employment = income stability = lower default risk |

---

### Engineered Features (Derived from Raw Inputs)

```python
loan_to_income        = loan_amnt / annual_inc           # affordability ratio
installment_to_income = installment / annual_inc         # monthly burden ratio
credit_utilization    = revol_bal / (revol_bal + bc_open_to_buy)
high_dti_flag         = 1 if dti > 20 else 0             # binary risk flag
low_fico_flag         = 1 if fico < 600 else 0           # binary risk flag
payment_capacity      = annual_inc - (installment × 12)  # free cash flow
credit_stress         = dti × loan_amnt                  # combined stress indicator
recent_inquiries_flag = 1 if inq_last_6mths > 3 else 0  # credit-seeking behavior
```

---

### Could Other Relevant Parameters Be Used?

Yes! These would further improve the model:

- **Loan-to-Value (LTV)** — if the loan is secured by a physical asset
- **Savings / Net Worth** — financial buffer against default
- **Payment history trend** — not just delinquencies but whether they're improving
- **Employment sector** — government jobs vs startups (stability differs)
- **Age of credit history** — longer history = more trustworthy borrower
- **Real-time macroeconomic indicators** — GDP growth, live unemployment rate
  *(your code currently uses hardcoded constants — a known limitation)*

---

## ❓ Q2: What is the Logic Behind Prediction — How is the Probability Calculated and the Threshold Set?

### Step-by-Step Prediction Logic (from `app.py`)

```
Step 1: User submits the loan form
         ↓
Step 2: preprocess_input(form_data)
         → Converts form fields into a 1-row DataFrame
         → Fills all numeric fields (loan_amnt, int_rate, etc.)
         → One-hot encodes categoricals (grade_A, term_36_months, etc.)
         ↓
Step 3: create_features_live(input_df)
         → Adds derived features: loan_to_income, installment_to_income,
           credit_utilization, high_dti_flag, low_fico_flag
         ↓
Step 4: add_economic_features(input_df)
         → Adds: inflation_rate=0.06, unemployment_rate=0.07, etc.
         ↓
Step 5: input_df.reindex(columns=MODEL_FEATURES, fill_value=0.0)
         → Aligns columns exactly to what the model was trained on
         ↓
Step 6: MODEL.predict_proba(input_df)[0][1]
         → Returns [prob_repay, prob_default]
         → We take index [1] = probability of DEFAULT
         ↓
Step 7: Apply threshold → assign verdict + risk label
         → Calculate LGD, EAD, Expected Loss
         ↓
Step 8: Save to history, run SHAP explanation, check drift
```

---

### How `predict_proba()` Works Internally (XGBoost)

XGBoost is an **ensemble of decision trees**. Each tree votes, and the votes are combined:

1. Internally works in **log-odds (logit)** space
2. Each tree outputs a small correction to the log-odds score
3. The final cumulative score is converted to a probability using the **sigmoid function**:

```
final_score = tree_1_output + tree_2_output + ... + tree_N_output
probability = 1 / (1 + e^(-final_score))
```

So if the trees collectively say "this person is risky" → `final_score` is large → sigmoid output is close to **1.0** (high default probability).

---

### How the Threshold (0.4) Was Set

During training, the optimal threshold was found mathematically:

```python
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
best_threshold = thresholds[(tpr - fpr).argmax()]   # Youden's J statistic
```

**Youden's J** picks the threshold that maximizes `(True Positive Rate − False Positive Rate)`.

However, in the actual prediction route, the threshold is **hardcoded as a business decision**:

```python
threshold = 0.4   # app.py line 452
```

Lowering the threshold → catches more defaulters (higher recall) but rejects more good customers.  
Raising the threshold → approves more people but misses more actual defaulters.

**In banking, it is safer to reject a good customer than to approve a defaulter** — hence 0.4 is a conservative choice.

---

## ❓ Q3: What Variables Were Used to Train the Model, What is the Accuracy, and Why is There "Default" in the Confusion Matrix?

### Variables Used for Training

**Raw features from the LendingClub dataset:**
- **Numeric:** `loan_amnt`, `int_rate`, `installment`, `annual_inc`, `dti`, `fico_range_low`, `fico_range_high`, `open_acc`, `revol_bal`, `revol_util`, `total_acc`, `delinq_2yrs`, `inq_last_6mths`, `pub_rec`, `pub_rec_bankruptcies`, `tax_liens`, `collections_12_mths_ex_med`, `acc_now_delinq`, `tot_coll_amt`, `tot_cur_bal`, `avg_cur_bal`, `bc_open_to_buy`, `bc_util`, `num_actv_bc_tl`, `num_rev_accts`, `percent_bc_gt_75`
- **Categorical (one-hot encoded):** `term`, `grade`, `sub_grade`, `emp_length`, `home_ownership`, `verification_status`, `purpose`, `addr_state`, `initial_list_status`
- **Engineered features:** `loan_to_income`, `installment_to_income`, `credit_utilization`, `payment_capacity`, `credit_stress`, `recent_inquiries_flag`, `high_dti_flag`, `low_fico_flag`
- **Target variable:** `loan_status` → 0 = Repay, 1 = Default

---

### Three Models Were Trained and Compared

| Model | Type | Selection Criteria |
|---|---|---|
| Logistic Regression | Linear baseline | — |
| Random Forest | Tree ensemble | — |
| **XGBoost** | Gradient boosted trees | ✅ **WINNER** (best profit score) |

The best model was selected based on **profit simulation** — not just accuracy — because catching a defaulter saves the full loan amount.

---

### Actual Model Metrics (from `model_metrics.json`)

| Metric | Value | What It Means |
|---|---|---|
| **Accuracy** | **68.68%** | Correct predictions out of all predictions |
| **Precision** | 34.78% | Of people flagged as default → only 35% actually default |
| **Recall** | 64.77% | Of actual defaulters → model catches 65% of them |
| **F1-Score** | 45.25% | Balanced measure of precision & recall |
| **ROC-AUC** | **74.06%** | Model's ability to rank defaulters above non-defaulters |

> **Note:** Accuracy alone is misleading here due to class imbalance. ROC-AUC (74%) is the more reliable metric.

---

### Confusion Matrix — Explained

```
                        Predicted: Repay    Predicted: Default
Actual: Repay (0)         TN = 16,719           FP = 7,284
Actual: Default (1)       FN =  2,113           TP = 3,884
```

| Cell | Full Name | Meaning in This Project |
|---|---|---|
| **TN = 16,719** | True Negative | Correctly predicted "Repay" → Good customers approved ✅ |
| **FP = 7,284** | False Positive | Predicted "Default" but actually repaid → Good customers wrongly rejected ⚠️ |
| **FN = 2,113** | False Negative | Predicted "Repay" but actually defaulted → **DANGEROUS — money lost!** ❌ |
| **TP = 3,884** | True Positive | Correctly predicted "Default" → Caught actual defaulters ✅ |

---

### Why Is There "Default" in the Confusion Matrix?

Because the **dataset is naturally imbalanced** — in real LendingClub data, approximately 80% of borrowers repay and only 20% default. Without correction, the model would simply predict "Repay" for everyone and still get 80% accuracy — which is useless.

Your project handles this imbalance with **two techniques**:

**1. SMOTE — Synthetic Minority Oversampling Technique:**
```python
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
# Creates synthetic "Default" samples to balance the training data
```

**2. XGBoost `scale_pos_weight`:**
```python
scale_pos_weight = counter[0] / counter[1]   # ≈ 4.0 if 4:1 imbalance
# Tells XGBoost to penalize missing a defaulter 4× more than missing a repayer
```

Despite these measures, FP (7,284) is still high — meaning the model aggressively flags potential defaulters. This is **intentional in banking** — it is safer to reject a good customer than to grant a loan to a defaulter.

---

## ❓ Q4: How is the Backend Working in This Project?

### Full Backend Architecture

```
Browser (User)
    │
    │  GET /
    ▼
Flask app.py ──► renders index.html (the loan assessment form)
    │
    │  POST /predict  (user submits form)
    ▼
┌─────────────────────────────────────────────────────────────┐
│                      predict() route                        │
│                                                             │
│  1.  Validate input (_validate_input)                       │
│      - loan_amnt ≥ $500                                     │
│      - annual_inc > 0                                       │
│      - 300 ≤ fico ≤ 850                                     │
│                                                             │
│  2.  preprocess_input()                                     │
│      - Parse form fields → 1-row DataFrame                  │
│      - Fill numerics, one-hot encode categoricals           │
│                                                             │
│  3.  create_features_live()                                 │
│      - Add derived/engineered features                      │
│                                                             │
│  4.  add_economic_features()                                │
│      - Add macro constants (inflation, unemployment, etc.)  │
│                                                             │
│  5.  Reindex to MODEL_FEATURES                              │
│      - Align columns exactly to training feature list       │
│                                                             │
│  6.  SHAP Explanation                                       │
│      - LoanModelExplainer.explain_single()                  │
│      - Returns top 5 features by |SHAP value|               │
│                                                             │
│  7.  MODEL.predict_proba(input_df)[0][1]                    │
│      - Returns probability of default (0.0 – 1.0)          │
│                                                             │
│  8.  Threshold logic → assign verdict                       │
│      - Calculate LGD, EAD, Expected Loss, Expected Profit   │
│                                                             │
│  9.  Build record dict → append to prediction_history.json  │
│                                                             │
│  10. log_decision() → governance audit log                  │
│                                                             │
│  11. Feedback loop check                                    │
│      - If history ≥ 100 entries → update training data      │
│                                                             │
│  12. Drift detection                                        │
│      - Compare current data distribution                    │
│        vs reference data (first 10,000 rows)                │
│      - If drift detected → trigger retraining               │
│                                                             │
│  13. retrain_model() (if triggered)                         │
│      - Runs: python -m src.train_model (subprocess)         │
│      - Reloads MODEL from disk                              │
│                                                             │
│  14. render_template("result.html")                         │
│      - Returns risk level, probability, verdict to browser  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Browser shows: Risk category, Default probability, Verdict
```

---

### Key Backend Components

| Component | File | Role |
|---|---|---|
| **Flask Web Server** | `webapp/app.py` | Routes, request handling, response rendering |
| **Trained Model** | `models/loan_default_model.pkl` | Serialized XGBoost model (saved with joblib) |
| **Feature List** | `utils/model_features.pkl` | Ordered list of column names the model expects |
| **Metrics** | `model_metrics.json` | Pre-computed accuracy, AUC, confusion matrix |
| **History** | `outputs/prediction_history.json` | All past predictions stored as JSON array |
| **SHAP Explainer** | `src/shap_explainer.py` | Explains which features drove each prediction |
| **Feedback Loop** | `feedback_loop.py` | Converts prediction history into new training data |
| **Drift Detection** | `monitoring/drift_detection.py` | Monitors for data distribution shifts |
| **Retraining** | `webapp/retrain.py` | Triggers `src.train_model` as a subprocess |
| **Governance Log** | `governance.py` | Logs every decision for audit and compliance |
| **Config** | `utils/config.py` | All paths, thresholds, and constants in one place |

---

### Model Loading at Flask Startup

```python
# When Flask starts, these are loaded once into memory:
MODEL          = joblib.load("models/loan_default_model.pkl")   # XGBoost model
SCALER         = joblib.load("models/scaler.pkl")               # (optional)
MODEL_FEATURES = pickle.load("utils/model_features.pkl")        # column names list
REFERENCE_DATA = pd.read_csv("data/processed/cleaned_data.csv").iloc[:10000]
EXPLAINER      = LoanModelExplainer()                           # SHAP engine
```

---

### Complete Data Flow — End to End

```
Raw CSV (LendingClub dataset)
         ↓
src/data_preprocessing.py   → clean, engineer features, encode categoricals
         ↓
src/train_model.py           → train XGBoost, save .pkl + features.pkl + metrics.json
         ↓
webapp/app.py                → load model at startup, serve predictions via Flask
         ↓
webapp/templates/result.html → show risk label and probability to user
         ↓
outputs/prediction_history.json → store every prediction result
         ↓
feedback_loop.py             → prediction history → new training rows
         ↓
retrain triggered            → model improves over time (continuous learning)
```

---

## 📋 Quick Summary for Viva (Revision Points)

| Question | Key Answer |
|---|---|
| **Algorithm used** | XGBoost (Gradient Boosted Trees) |
| **Why XGBoost** | Best profit score among Logistic Regression, Random Forest, XGBoost |
| **Target variable** | `loan_status` → 0 = Repay, 1 = Default |
| **Probability output** | `predict_proba()[0][1]` via sigmoid of tree ensemble |
| **Decision threshold** | 0.40 (business decision — conservative) |
| **Accuracy** | 68.68% |
| **ROC-AUC** | 74.06% (better metric for imbalanced data) |
| **Recall** | 64.77% (catches 65% of actual defaulters) |
| **Imbalance handling** | SMOTE + XGBoost `scale_pos_weight` |
| **Backend framework** | Flask (Python web framework) |
| **Explainability** | SHAP (TreeSHAP) — top 5 features by |SHAP value| |
| **Auto-retraining** | Triggered on drift detection or every 100 new predictions |
| **Most important features** | FICO score, DTI, int_rate, loan_amnt, delinq_2yrs |
