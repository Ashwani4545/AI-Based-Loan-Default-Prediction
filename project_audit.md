# 🔍 AI-Based Loan Default Prediction — Project Audit

---

## 1. 🚨 Problems Found in the Project

### 🔴 CRITICAL BUGS

#### BUG 1 — `credit_utilization` formula is WRONG (app.py line 217)
**File:** `webapp/app.py` → `create_features_live()`

```python
# WRONG — always returns ~1.0 regardless of input:
df["credit_utilization"] = df["revol_bal"] / (df["revol_bal"] + 1e-6)

# CORRECT — matches train_model.py formula:
df["credit_utilization"] = df["revol_bal"] / (df["revol_bal"] + df["bc_open_to_buy"] + 1e-6)
```
**Impact:** This is a **train/serve skew** — the feature is computed differently at training vs prediction time. The model was trained with `revol_bal / (revol_bal + bc_open_to_buy)` but predictions use `revol_bal / (revol_bal + 1e-6)` which is basically always `1.0`.

---

#### BUG 2 — `create_features_live()` is MISSING features that training creates
**File:** `webapp/app.py` vs `src/train_model.py`

Training (`train_model.py`) creates extra features not in `create_features_live()`:

| Feature in Training | In `create_features_live()`? |
|---|---|
| `loan_to_income` | ✅ |
| `installment_to_income` | ✅ |
| `credit_utilization` | ✅ (but wrong formula) |
| `high_dti_flag` | ✅ |
| `low_fico_flag` | ✅ |
| `payment_capacity` | ❌ **MISSING** |
| `credit_stress` | ❌ **MISSING** |
| `recent_inquiries_flag` | ❌ **MISSING** |

These are filled with `0.0` by `reindex(fill_value=0.0)` but that is not the same as computing them properly.

---

#### BUG 3 — Scaler applied on reindexed DataFrame (wrong column order)
**File:** `webapp/app.py` lines 428–442

```python
# SCALER was fit on training columns in a fixed order
# But before calling scaler.transform(), columns are reindexed via MODEL_FEATURES
# The reindex happens BEFORE scaling which is correct,
# BUT the scaler.pkl file is NOT SAVED in train_model.py at all!
```
`train_model.py` never saves a scaler — it uses SMOTE on raw/encoded data without StandardScaler.  
`app.py` tries to load `scaler.pkl` at startup but it won't exist → it gracefully falls back to "no scaler".  
This means the SCALER code path will never be used unless someone manually saves a scaler, which is misleading.

---

#### BUG 4 — Duplicate / inconsistent risk-classification functions
**File:** `webapp/app.py` — there are **4 different functions** doing the same thing with different thresholds:

| Function | LOW threshold | MEDIUM threshold | HIGH threshold |
|---|---|---|---|
| `get_risk_category()` | < 0.2 | < 0.4 | < 0.6 |
| `get_risk_info()` | < 0.3 | < 0.6 | else |
| `get_decision()` | < 0.3 | < 0.6 | else |
| `credit_policy()` | (returns approval strings, not risk) | | |
| `config.py RISK_LEVELS` | < 0.30 | < 0.50 | < 0.70 |
| `predict()` inline logic | < 0.4 | < 0.6 | else |

**None of them are consistent.** The prediction route uses the *inline* logic and ignores `get_risk_level()` from config.

---

#### BUG 5 — `policy_decision` contradicts `verdict`
**File:** `webapp/app.py` lines 488–489

```python
prediction = verdict                     # "Repay", "Review", or "Default"
policy_decision = "Default" if prob > threshold else "Repay"  # threshold=0.4
```
At `prob = 0.45` → `verdict = "Review"` but `policy_decision = "Default"`. These two fields with contradictory logic are both stored in the history, causing confusion.

---

#### BUG 6 — Feedback loop retraining happens on EVERY prediction (not just multiples of 100)
**File:** `webapp/app.py` lines 558–564

```python
feedback_data = build_feedback_dataset()
if feedback_data is not None:        # True whenever history >= 100 entries
    update_training_data(feedback_data)
    retrain_model()                  # Called on EVERY prediction after 100 entries!
    reload_model()
```
`build_feedback_dataset()` returns data when `len(history) >= 100`. This means after 100 predictions, `retrain_model()` is called on **every single request**, making the app extremely slow.  
`should_retrain()` (line 166) already has the correct `% 100 == 0` logic but it's called **after** the always-retrain block.

---

#### BUG 7 — `save_report()` uses a **relative** path
**File:** `webapp/app.py` lines 356–361

```python
def save_report(report, record_id):
    path = f"reports/{record_id}.txt"   # ← relative path, breaks when CWD changes
```
This will fail unless Flask is started from the project root. Should use an absolute path.

---

### 🟡 LOGIC / DESIGN PROBLEMS

#### PROBLEM 8 — Alternative features are RANDOM during training, then expected from form
**File:** `src/train_model.py` lines 144–147

```python
df["mobile_usage_score"]      = np.random.randint(1, 100, len(df))
df["digital_txn_count"]       = np.random.randint(1, 50, len(df))
df["utility_payment_score"]   = np.random.randint(1, 100, len(df))
df["employment_stability"]    = np.random.randint(1, 10, len(df))
```
The model is trained on **random noise** for these columns. The form does NOT collect them from users. So for any real prediction they are `0`, which is inconsistent with training distribution. These features are meaningless and pollute the model.

---

#### PROBLEM 9 — Economic features are hardcoded constants
**File:** `src/train_model.py` lines 157–164 and `webapp/app.py` lines 224–233

```python
df["inflation_rate"]    = 0.06   # same for every row
df["interest_rate_env"] = 0.08   # same for every row
df["unemployment_rate"] = 0.07   # same for every row
```
Since these are the same for every training sample, **the model learns zero information from them**. At prediction time the same constants are added again — they do nothing and add unnecessary computation.

---

#### PROBLEM 10 — `check_group_bias()` uses `gender` which is never in the form
**File:** `src/shap_explainer.py` lines 221–228

```python
gender = input_data.get("gender", None)
if gender == "Female" and income < 20000:
    return "⚠️ Potential bias risk..."
```
The form (`index.html`) has NO `gender` field. So `gender` is always `None` and this check never triggers. Similarly, `validate_sensitive_features()` looks for `gender`, `race`, `religion` — also never in the form.

---

#### PROBLEM 11 — `get_current_data()` has a column rename bug
**File:** `webapp/app.py` lines 181–186

```python
cols = ["loan_amnt", "int_rate", "installment", "annual_inc",
        "dti", "fico", "open_acc", "revol_bal", "total_acc"]
df = df.rename(columns={"fico": "fico_range_low"})
return df[cols].dropna()   # ← "fico" is renamed but cols still uses "fico" → KeyError!
```
This will throw a `KeyError` because `cols` still references `"fico"` (which was renamed away).

---

#### PROBLEM 12 — `feedback_loop.py` has a column mismatch bug
**File:** `feedback_loop.py` lines 25–32

```python
features = ["loan_amnt", "int_rate", "annual_inc", "fico",
            "dti", "open_acc", "revol_bal", "total_acc"]
df = df.rename(columns={"fico": "fico_range_low"})
df = df[features + ["prediction"]].dropna()   # ← "fico" is in features but renamed → KeyError!
```
Same rename-then-select bug. Should be `df[["loan_amnt", "int_rate", "annual_inc", "fico_range_low", ...]]` after renaming.

---

#### PROBLEM 13 — `result.html` does NOT show SHAP explanation
The result page only shows: probability, risk, warning, and verdict.
Despite the code computing SHAP explanations and storing `top_features` in the record, **none of this is passed to `result.html`**.

The `render_template()` call:
```python
return render_template("result.html",
    risk=risk_label,
    show_warning=show_warning,
    prob=prob,
    verdict=verdict,
)
```
Missing: `top_features`, `expected_loss`, `borrower`, `loan_amnt`, `explanation`, etc.

---

---

## 2. 📊 Prediction Logic: Is it Correct?

### Flow
```
Form Input → preprocess_input() → create_features_live() → add_economic_features()
→ reindex to MODEL_FEATURES → SHAP explanation → MODEL.predict_proba() → threshold → verdict
```

### Decision Thresholds (current inline logic in predict()):
| Probability | Risk Label | Verdict |
|---|---|---|
| prob ≤ 0.4 | LOW RISK | Repay (No default) |
| 0.4 < prob ≤ 0.6 | MEDIUM RISK | Review |
| prob > 0.6 | HIGH RISK | Default |
| loan > 5× income (any prob) | HIGH RISK (Override) | Override triggered |

### Issues with Prediction Logic:
1. **Threshold inconsistency** — The config has LOW=0.30, but prediction route uses 0.4 as the "low" boundary.
2. **"Review" verdict** — probabilities 40–60% return "Review" not "Default". This is a business decision, okay if intentional.
3. **Override logic** — if loan > 5× annual income, it's always "High Risk" regardless of model output. This is reasonable but is not documented.
4. **Train/serve skew** — as described in BUGS 1 & 2, the features fed to the model at prediction time differ from what was used at training. This can cause unreliable probabilities.

---

## 3. 📝 Form Fields Analysis — Are They All Needed?

### Star-marked (required `*`) fields:
| Field | Required in Form | Used by Model | Verdict |
|---|---|---|---|
| `loan_amnt` | ✅ | ✅ Yes | **Keep — Required** |
| `term` | ✅ | ✅ One-hot encoded | **Keep — Required** |
| `int_rate` | ✅ | ✅ Yes | **Keep — Required** |
| `annual_inc` | ✅ | ✅ Yes | **Keep — Required** |
| `fico_range_low` | ✅ | ✅ Yes | **Keep — Required** |
| `grade` | ✅ | ✅ One-hot encoded | **Keep — Required** |
| `purpose` | ✅ | ✅ One-hot encoded | **Keep — Required** |

### Non-starred (optional) fields:
| Field | In Form | Used by Model | Verdict |
|---|---|---|---|
| `installment` | ✅ | ✅ Used in `installment_to_income` ratio | **Keep** |
| `initial_list_status` | ✅ | ✅ One-hot encoded | **Keep** |
| `dti` | ✅ | ✅ Used in `high_dti_flag`, `credit_stress` | **Keep** |
| `fico_range_high` | ✅ | ✅ Used (model feature) | **Keep** |
| `revol_bal` | ✅ | ✅ Used in credit utilization | **Keep** |
| `revol_util` | ✅ | ✅ Model feature | **Keep** |
| `open_acc` | ✅ | ✅ Model feature | **Keep** |
| `total_acc` | ✅ | ✅ Model feature | **Keep** |
| `delinq_2yrs` | ✅ | ✅ Model feature | **Keep** |
| `inq_last_6mths` | ✅ | ✅ Used in `recent_inquiries_flag` | **Keep** |
| `pub_rec` | ✅ | ✅ Model feature | **Keep** |
| `pub_rec_bankruptcies` | ✅ | ✅ Model feature | **Keep** |
| `tax_liens` | ✅ | ✅ Model feature | **Keep** |
| `collections_12_mths_ex_med` | ✅ | ✅ Model feature | **Keep** |
| `acc_now_delinq` | ✅ | ✅ Model feature | **Keep** |
| `tot_coll_amt` | ✅ | ✅ Model feature | **Keep** |
| `tot_cur_bal` | ✅ | ✅ Model feature | **Keep** |
| `avg_cur_bal` | ✅ | ✅ Model feature | **Keep** |
| `bc_open_to_buy` | ✅ | ✅ Used in credit_utilization | **Keep** |
| `bc_util` | ✅ | ✅ Model feature | **Keep** |
| `num_actv_bc_tl` | ✅ | ✅ Model feature | **Keep** |
| `num_rev_accts` | ✅ | ✅ Model feature | **Keep** |
| `percent_bc_gt_75` | ✅ | ✅ Model feature | **Keep** |
| `sub_grade` | ✅ | ✅ One-hot encoded | **Keep** |
| `emp_length` | ✅ | ✅ One-hot encoded | **Keep** |
| `home_ownership` | ✅ | ✅ One-hot encoded | **Keep** |
| `verification_status` | ✅ | ✅ One-hot encoded | **Keep** |
| `addr_state` | ✅ | ✅ One-hot encoded | **Keep** |

### Fields MISSING from the form (collected by model but not in form):
| Feature | How filled in prediction | Is it a Problem? |
|---|---|---|
| `earliest_cr_line` | Not filled (defaults 0) | ⚠️ Listed in `_CATEGORICAL_FIELDS` but no form input |
| `mobile_usage_score` | Defaults to 0 | ⚠️ Model trained on random 1–100, but gets 0 at inference |
| `digital_txn_count` | Defaults to 0 | ⚠️ Same issue |
| `utility_payment_score` | Defaults to 0 | ⚠️ Same issue |
| `employment_stability` | Defaults to 0 | ⚠️ Same issue |

### Conclusion:
All starred fields ARE needed. No field in the form is useless — they all feed into the model.  
However, **4 alternative data fields** (`mobile_usage_score`, `digital_txn_count`, `utility_payment_score`, `employment_stability`) are trained on random data and never collected from users — these should either be **added to the form** or **removed from training**.

---

## 4. 🔬 How SHAP is Explaining Predictions

### Current SHAP Flow:
```
LoanModelExplainer.__init__()
  → loads model (separate instance from MODEL in app.py!)
  → initialises shap.Explainer(self.model)

predict() calls EXPLAINER.explain_single(input_df)
  → shap_values = self.explainer(input_df)     # TreeExplainer internally
  → importance = |shap_values.values[0]|       # absolute SHAP for each feature
  → top 5 highest |SHAP| features returned
```

### What SHAP is Actually Doing:
- Uses **TreeSHAP** (SHAP's `Explainer` auto-selects TreeExplainer for XGBoost)
- SHAP values represent the **marginal contribution** of each feature to the model's output (in log-odds space)
- `|shap_value|` = magnitude of feature's push toward/away from default
- The top 5 `{feature, impact}` pairs are returned but currently only stored in history, **not shown on the result page**

### Problems with SHAP Implementation:

1. **Two separate model instances** — `EXPLAINER` loads the model independently from the global `MODEL`. After retraining and `reload_model()`, the global `MODEL` is updated but `EXPLAINER.model` is **stale** (still the old model). SHAP explanations and predictions use different models.

2. **SHAP output not shown to users** — `top_features` is computed and stored in history but `result.html` does not render it. Users never see the explanation.

3. **SHAP uses `input_df` BEFORE reindexing to `MODEL_FEATURES`** — See `app.py` line 432 vs 429:
   ```python
   explanation = EXPLAINER.explain_single(input_df)   # ← before reindex
   # ... 
   input_df = input_df.reindex(columns=columns, fill_value=0.0)  # reindex happens AFTER
   ```
   So SHAP receives a DataFrame that may not match the model's expected feature columns exactly, whereas the actual model receives the reindexed version. This can produce incorrect SHAP attributions.

4. **Fallback uses global feature importances, not per-prediction SHAP** — If SHAP is not installed, `_fallback_importances()` returns the same global importances for every prediction, regardless of the input.

---

## 5. 📋 Summary Table of All Problems

| # | Severity | File | Problem |
|---|---|---|---|
| 1 | 🔴 Critical | `app.py` | `credit_utilization` formula wrong (train/serve skew) |
| 2 | 🔴 Critical | `app.py` | 3 engineered features missing at inference time |
| 3 | 🟡 Medium | `app.py` | Scaler code is dead code (scaler never saved) |
| 4 | 🟡 Medium | `app.py` | 4 duplicate, inconsistent risk classification functions |
| 5 | 🟡 Medium | `app.py` | `policy_decision` contradicts `verdict` |
| 6 | 🔴 Critical | `app.py` | Retraining fires on EVERY prediction (not just ÷100) |
| 7 | 🟡 Medium | `app.py` | `save_report()` uses relative path |
| 8 | 🟡 Medium | `train_model.py` | Alternative features trained on random noise |
| 9 | 🟢 Low | `train_model.py` + `app.py` | Economic features are hardcoded constants (no signal) |
| 10 | 🟢 Low | `shap_explainer.py` | `check_group_bias()` checks `gender` which is never collected |
| 11 | 🔴 Critical | `app.py` | `get_current_data()` KeyError — renames then selects old name |
| 12 | 🔴 Critical | `feedback_loop.py` | Same rename-then-select KeyError bug |
| 13 | 🟡 Medium | `result.html` | SHAP explanation computed but NEVER shown to users |
| 14 | 🟡 Medium | `shap_explainer.py` | SHAP uses stale model after retraining |
| 15 | 🟡 Medium | `app.py` line 432 | SHAP called BEFORE `reindex` — different columns than model sees |
| 16 | 🟢 Low | form | `earliest_cr_line` in `_CATEGORICAL_FIELDS` but not in form |
| 17 | 🟢 Low | Training | Alternative data fields need form inputs or should be removed |
