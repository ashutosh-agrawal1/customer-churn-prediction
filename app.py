import streamlit as st
import numpy as np
import pickle

with open("churn_model.pkl", "rb") as f:
    data = pickle.load(f)

pipeline = data["pipeline"]


# -----------------------------
# App UI
# -----------------------------
st.set_page_config(layout="centered",page_title="Churn Predictor")
st.title("📊 Customer Churn Prediction")
st.markdown("Predict whether a customer is likely to churn based on key features.")

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("🔍 Enter Customer Details")
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)
# -----------------------------
# Feature Encoding
# -----------------------------

contract_map = {
    "Month-to-month": [1, 0, 0],
    "One year": [0, 1, 0],
    "Two year": [0, 0, 1]
}
contract_encoded = contract_map[contract]

input_data = np.array([[
    tenure,
    monthly_charges,
    *contract_encoded
]])


# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Churn"):

    prob = pipeline.predict_proba(input_data)[0][1]

    st.subheader(f"📈 Churn Probability: {prob:.2f}")

    # -----------------------------
    # Risk Level
    # -----------------------------
    if prob > 0.7:
        st.error("🔴 High Risk Customer")
    elif prob > 0.4:
        st.warning("🟡 Medium Risk Customer")
    else:
        st.success("🟢 Low Risk Customer")

    # -----------------------------
    # 💡 Business Explanation
    # -----------------------------
    st.markdown("### 💡 Why this prediction?")

    reasons = []

    # ---- Feature-based reasoning ----
    if tenure < 12:
        reasons.append("Low tenure (new customers are more likely to churn)")
    elif tenure > 36:
        reasons.append("High tenure (loyal customers are less likely to churn)")

    if monthly_charges > 70:
        reasons.append("High monthly charges (price-sensitive customers may leave)")
    elif monthly_charges < 30:
        reasons.append("Low monthly charges (lower financial pressure)")

    if contract == "Month-to-month":
        reasons.append("Flexible contract (higher churn risk)")
    elif contract == "One year":
        reasons.append("Moderate commitment contract")
    elif contract == "Two year":
        reasons.append("Long-term contract (lower churn risk)")

    # ---- Display reasons ----
    if reasons:
        for r in reasons:
            st.write(f"• {r}")
    else:
        st.write("Customer profile appears stable with no strong churn indicators.")

    # -----------------------------
    # 📌 Recommended Action
    # -----------------------------
    st.markdown("### 📌 Recommended Action")

    if prob > 0.7:
        st.error("🚨 Immediate action required")

        if contract == "Month-to-month":
            st.write("👉 Offer discount to switch to long-term plan")

        if tenure < 6:
            st.write("👉 Improve onboarding experience for new customer")

        if monthly_charges > 70:
            st.write("👉 Provide pricing incentives or bundle offers")

    elif prob > 0.4:
        st.warning("⚠️ Moderate risk — monitor closely")

        st.write("👉 Engage customer with personalized offers")
        st.write("👉 Send retention emails or loyalty benefits")

    else:
        st.success("✅ Low risk — stable customer")

        st.write("👉 Upsell premium services")
        st.write("👉 Offer loyalty rewards to increase lifetime value")