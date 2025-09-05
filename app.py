import streamlit as st
import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier

# --- Save trained model (run this after training in your notebook) ---
# Uncomment and run this block in your notebook after fitting xgb_best
# import pickle
# pickle.dump(xgb_best, open("xgb_best_model.pkl", "wb"))

st.title("Insolvency Prediction App")

# Load trained model
model = pickle.load(open("xgb_best_model.pkl", "rb"))

st.header("Enter Company Financial Data")

# Example: Replace with your actual feature names
feature_names = [
    # Add all feature names used in your model here
    " Net worth/Assets", " Debt ratio %", " Gross Profit to Sales", " Operating Gross Margin", " Net Value Per Share (C)",
    " Net Value Per Share (A)", " Realized Sales Gross Margin", " Net Value Per Share (B)", " Operating profit/Paid-in capital",
    " Operating Profit Per Share (Yuan ¥)", " Regular Net Profit Growth Rate", " After-tax Net Profit Growth Rate",
    " ROA(B) before interest and depreciation after tax", " ROA(C) before interest and depreciation before interest",
    " Liability to Equity", " Current Liabilities/Equity", " Current Liability to Equity", " Net profit before tax/Paid-in capital",
    " Per Share Net profit before tax (Yuan ¥)", " Net Income to Total Assets", " ROA(A) before interest and % after tax",
    " Persistent EPS in the Last Four Seasons", " Borrowing dependency", " Operating Funds to Liability", " Cash flow rate",
    " Current Liability to Assets", " Equity to Long-term Liability", " Net Income to Stockholder's Equity",
    " Retained Earnings to Total Assets", " Working Capital/Equity", " Contingent liabilities/Net worth", " Total Asset Turnover",
    " Current Assets/Total Assets", " Quick Assets/Total Assets", " CFO to Assets", " Cash Reinvestment %", " Cash Flow Per Share",
    " Working Capital to Total Assets", " Cash Flow to Liability", " Cash Flow to Total Assets",
    " Inventory and accounts receivable/Net value"
]

user_input = []
for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0)
    user_input.append(value)

if st.button("Predict Insolvency"):
    X_input = np.array([user_input])
    prediction = model.predict(X_input)
    proba = model.predict_proba(X_input)[0][1]
    st.write("Prediction:", "Bankrupt" if prediction[0] == 1 else "Solvent")
    st.write(f"Probability of Bankruptcy: {proba:.2f}")
