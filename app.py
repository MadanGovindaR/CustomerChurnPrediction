import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("customer_churn_model.pkl", "rb"))

st.title("Customer Churn Prediction System")

# User Inputs
age = st.number_input("Enter Age")
monthly_bill = st.number_input("Enter Monthly Bill")
contract_years = st.number_input("Enter Contract Years")
support_calls = st.number_input("Enter Support Calls")
internet_usage = st.number_input("Enter Internet Usage")

# Prediction Button
if st.button("Predict"):

    input_data = np.array([
        [age, monthly_bill, contract_years, support_calls, internet_usage]
    ])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Customer Will Leave")
    else:
        st.success("Customer Will Stay")