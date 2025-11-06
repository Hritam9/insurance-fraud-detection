import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Insurance Fraud Scoring Dashboard", layout="centered")
st.title("🏦 Insurance Claim Fraud Scoring")

model = joblib.load("models/fraud_model.pkl")

claim_amount = st.number_input("Claim Amount", 1000, 100000, 15000)
region_risk = st.selectbox("Region Risk", ["Low", "Medium", "High"])
claim_type = st.selectbox("Claim Type", ["Vehicle", "Health", "Home"])
customer_age = st.slider("Customer Age", 18, 70, 35)
past_claims = st.number_input("Past Claims", 0, 10, 1)

region_map = {"Low": 0, "Medium": 1, "High": 2}
claim_map = {"Vehicle": 0, "Health": 1, "Home": 2}

X = pd.DataFrame([{
    "claim_amount": claim_amount,
    "region_risk_score": region_map[region_risk],
    "claim_type_code": claim_map[claim_type],
    "customer_age": customer_age,
    "past_claims": past_claims
}])

if st.button("Predict Fraud Probability"):
    pred = model.predict_proba(X)[:, 1][0]
    st.metric("Fraud Probability", f"{pred:.2%}")
