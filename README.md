# AI Loan Default Prediction System

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
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict.py
│
├── models/
│   └── loan_default_model.pkl
│
├── explainability/
│   └── shap_explainer.py
│
├── monitoring/
│   └── drift_detection.py
│
├── webapp/
│   ├── app.py                → Main Flask backend
│   ├── templates/            → HTML pages (UI)
│   │    ├── index.html        → Input form (user enters data)
│   │    ├── result.html       → Prediction result page
│   │    └── dashboard.html    → Admin/analytics page
│   ├── static/               → Frontend assets
│       ├── css/style.css     → Styling
│       └── js/script.js      → JS logic (optional)
│
├── utils/
│   └── config.py
│
├── requirements.txt
│
└── README.md
```
------------------------------------------------------------------------

## Installation

Clone the repository

git clone https://github.com/yourusername/AI-Loan-Default-Prediction.git

Navigate to project folder

cd AI-Loan-Default-Prediction

Install dependencies

pip install -r requirements.txt

------------------------------------------------------------------------

## Running the Application

Start the Flask server

`python webapp/app.py`

Open your browser

`http://127.0.0.1:5000`

Enter borrower information to receive loan default prediction results.

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
