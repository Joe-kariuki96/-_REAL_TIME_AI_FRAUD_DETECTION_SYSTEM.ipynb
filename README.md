# 🧠 Real-Time AI Fraud Detection System
End-to-End Machine Learning Pipeline for Financial Transactions

Dataset → EDA → Feature Engineering → Modeling → Evaluation → Deployment Prep

## 📌 Project Overview

This project builds a complete real-time fraud detection system using a synthetic yet realistic dataset of financial transactions.
The workflow mirrors what is used in modern fintech companies like Stripe, PayPal, Revolut, Chime, and Visa.

The goal is to develop a machine-learning model capable of identifying suspicious transactions based on behavioral, geographic, and risk-related features.

## 🚀 Key Features
✔ Synthetic but realistic dataset (150,000+ transactions)

Replicates real-world fraud patterns including IP risk, location mismatch, device behavior, and transaction anomalies.

## ✔ Full Exploratory Data Analysis (EDA)

Visualizes:

Fraud distribution

Transaction amount patterns

IP risk score behavior

Country mismatch trends

Hour-of-day fraud spikes

## ✔ Feature Engineering

Includes:

One-hot encoding

Behavioral features

Geographic risk indicators

Time-based patterns

## ✔ Imbalanced Learning with SMOTE

Balances fraud vs. non-fraud to improve model recall and detection quality.

## ✔ Model Training & Comparison

Models trained:

Logistic Regression

XGBoost

LightGBM

LightGBM performed the best with the highest AUC score.

## ✔ Model Evaluation

Includes:

AUC score comparison

Confusion matrix

Precision, recall, F1-score

Feature importance visualization

## ✔ Deployment-Ready Artifacts

Saved:

fraud_model.pkl

scaler.pkl

These are ready for:

FastAPI real-time inference

Streamlit monitoring dashboard

## 🧪 Technologies Used
Category	Tools
Programming	Python
ML Libraries	scikit-learn, XGBoost, LightGBM
Data	Pandas, NumPy
Visualization	Seaborn, Matplotlib
Imbalanced Learning	SMOTE (imblearn)
Model Serving	FastAPI (optional)
Dashboard	Streamlit (optional)
Persistence	joblib
📊 Model Performance (Summary)

LightGBM AUC: ~0.75

XGBoost AUC: ~0.75

Logistic Regression AUC: ~0.69

LightGBM & XGBoost captured the complex non-linear patterns of fraud much better than Logistic Regression.

## 🔍 Classification Report Highlights

Good precision for non-fraud

Moderate recall for fraud

Balanced F1-score

Clear improvement from SMOTE

## 🔥 Feature Importance (Top Predictors)

IP risk score

Hour of day

Country mismatch

Device behavior

Past fraud count

Transaction amount

These align strongly with real-world fraud detection patterns.

## 📁 Project Structure

fraud-detection-system/
│

├── data/
│   └── transactions.csv
│

├── notebooks/
│   └── real_time_ai_fraud_detection_system.ipynb
│
├── model/
│   ├── fraud_model.pkl
│   └── scaler.pkl
│

├── api/
│   └── (FastAPI service - optional)
│

├── dashboard/
│   └── (Streamlit dashboard - optional)
│

└── README.md

## 🚀 Next Steps (Optional Enhancements)

Build FastAPI endpoint for real-time fraud scoring

Build Streamlit dashboard for fraud monitoring

Integrate SHAP explainability

Deploy model to Render, Railway, or AWS

Add CI/CD for retraining and version control
