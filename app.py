import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Title
st.title("Fintech Microloan Credit Scoring Demo")

# Collect user input
st.sidebar.header("📋 Borrower Details")
age = st.sidebar.slider("Age", 18, 60, 25)
monthly_income = st.sidebar.number_input("Monthly Income (₹)", 1000, 100000, 15000)
employment_type = st.sidebar.selectbox("Employment Type", ["Salaried", "Self-Employed", "Student"])
existing_loans = st.sidebar.slider("Existing Loans", 0, 5, 0)
education = st.sidebar.selectbox("Education Level", ["High School", "Graduate", "Post Graduate"])
transaction_score = st.sidebar.slider("Online Transaction Score (0–100)", 0, 100, 50)

# Convert categorical values to numeric
employment_map = {"Salaried": 1, "Self-Employed": 2, "Student": 3}
education_map = {"High School": 1, "Graduate": 2, "Post Graduate": 3}

X_input = pd.DataFrame([[
    age,
    monthly_income,
    employment_map[employment_type],
    existing_loans,
    education_map[education],
    transaction_score
]], columns=["age", "income", "employment", "loans", "education", "txn_score"])

# Dummy training data (for demo only)
X_train = pd.DataFrame([
    [22, 15000, 3, 0, 1, 50],
    [35, 30000, 1, 1, 2, 70],
    [45, 50000, 2, 2, 2, 40],
    [28, 18000, 3, 0, 1, 80],
    [50, 60000, 1, 0, 3, 90]
], columns=X_input.columns)

y_train = [0, 1, 1, 0, 1]  # 1 = likely to repay, 0 = not likely

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Prediction
prediction = model.predict(X_input)[0]
score = model.predict_proba(X_input)[0][1] * 100

# Output
st.subheader("📊 Prediction Result")
if prediction == 1:
    st.success(f"✅ Approved: Likely to Repay ({score:.2f}% confidence)")
else:
    st.error(f"❌ Rejected: Risk of Default ({score:.2f}% confidence)")