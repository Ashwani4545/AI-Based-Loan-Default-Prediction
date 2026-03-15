# ==========================================
# Loan Default Prediction
# Data Preprocessing + Feature Engineering
# ==========================================

# Import libraries
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE


# ------------------------------------------
# 1. Define Project Paths
# ------------------------------------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
data_path = os.path.join(BASE_DIR, "data", "raw", "loan_dataset.csv")

print("Looking for dataset at:", data_path)

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at: {data_path}")


# ------------------------------------------
# 2. Load Dataset
# ------------------------------------------

df = pd.read_csv(data_path)

print("Dataset Loaded Successfully")
print("Dataset Shape:", df.shape)


# ------------------------------------------
# 3. Basic Data Cleaning
# ------------------------------------------

df = df.drop_duplicates()

num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# Fill missing numerical values
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill missing categorical values
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


# ------------------------------------------
# 4. Feature Engineering
# ------------------------------------------

print("\nPerforming Feature Engineering...")

# Example: Debt to Income Ratio
if "loan_amount" in df.columns and "income" in df.columns:
    df["debt_to_income_ratio"] = df["loan_amount"] / (df["income"] + 1)

# Example: Loan to Income Ratio
if "loan_amount" in df.columns and "income" in df.columns:
    df["loan_income_ratio"] = df["loan_amount"] / (df["income"] + 1)

# Example: Credit Utilization
if "credit_limit" in df.columns and "credit_used" in df.columns:
    df["credit_utilization"] = df["credit_used"] / (df["credit_limit"] + 1)

# Example: Employment Stability Feature
if "years_employed" in df.columns:
    df["employment_stability"] = df["years_employed"] / (df["age"] + 1)

# Example: Income Category
if "income" in df.columns:
    df["income_category"] = pd.cut(
        df["income"],
        bins=[0, 30000, 60000, 100000, np.inf],
        labels=[0, 1, 2, 3]
    )


# ------------------------------------------
# 5. Encode Categorical Variables
# ------------------------------------------

label_encoder = LabelEncoder()

for col in cat_cols:
    df[col] = label_encoder.fit_transform(df[col])


# ------------------------------------------
# 6. Define Features and Target
# ------------------------------------------

target_column = "loan_status"

if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found")

X = df.drop(target_column, axis=1)
y = df[target_column]


# ------------------------------------------
# 7. Train-Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train Size:", X_train.shape)
print("Test Size:", X_test.shape)


# ------------------------------------------
# 8. Feature Scaling
# ------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ------------------------------------------
# 9. Handle Class Imbalance
# ------------------------------------------

smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# ------------------------------------------
# 10. Final Output
# ------------------------------------------

print("\nPreprocessing Completed Successfully")

print("Original Training Shape:", X_train.shape)
print("Resampled Training Shape:", X_train_resampled.shape)
print("Testing Shape:", X_test.shape)