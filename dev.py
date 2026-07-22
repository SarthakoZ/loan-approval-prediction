import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("Loan Approval Prediction")

with st.sidebar:
    st.title("Loan Predictor")
    name = st.text_input("Enter your name:")
    st.write(f"• Welcome {name}")
    loan_type = st.selectbox("Select which type of loan you are applying for : ", ["Select > ", "Home Loan", "Personal Loan", "Education Loan", "Vehicle Loan", "Business Loan"])
    st.write(f"• You are applying for {loan_type}")
    st.subheader("• Description: ")
    st.write("This Machine Learning model predicts whether a loan application is likely to be approved or rejected based on the applicant's details.")
    st.write("• Selected Model : Random Forest Classifier")
    st.write("• Accuracy : 98.13%")

st.write("Enter the applicant details below and click **Predict**.")

model = joblib.load("loan_approval_final_model.pkl")

st.subheader("Applicant Details")

loan_amount = st.number_input("Loan Amount (₹)")

col1, col2 = st.columns(2)

with col1:
    no_of_dependents = st.slider("Dependents", 0, 10, 0)
    self_employed = st.selectbox("Self Employed", ["no", "yes"])
    education = st.selectbox("Education", ["graduate", "not graduate"])
    income_annum = st.number_input("Annual Income (₹)", min_value=0)
    residential_assets_value = st.number_input("Residential Assets (₹)", min_value=0)

with col2:
    loan_term = st.slider("Loan Term (Years)", 1, 30, 1)
    cibil_score = st.slider("CIBIL Score", 300, 900, 300)
    commercial_assets_value = st.number_input("Commercial Assets (₹)", min_value=0)
    luxury_assets_value = st.number_input("Luxury Assets (₹)", min_value=0)
    bank_asset_value = st.number_input("Bank Assets (₹)", min_value=0)

if st.button("🔍 Predict Loan Status"):

    data = pd.DataFrame([{
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value
    }])

    with st.spinner("Analyzing application..."):
        pred = model.predict(data)[0]

    st.divider()

    if pred == 0:
        st.success(f"Congratulations {name}! Loan Status: APPROVED")
        st.balloons()

    elif pred == 1:
        st.error(f"Sorry {name}! Loan Status: REJECTED")

    else:
        st.warning("Unexpected prediction result.")

    with st.expander("View Submitted Details"):
        st.dataframe(data)

st.write("## Model Selection and Comparison")

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "KNN",
        "Decision Tree",
        "Random Forest",
        "SVC"
    ],
    "Accuracy": [
        "92.5%",
        "90.5%",
        "97.18%",
        "98.13%",
        "94.14%"
    ],
    "Status": [
        "Evaluated",
        "Evaluated",
        "Evaluated",
        "Selected",
        "Evaluated"
    ]
})

st.table(comparison)

st.info(
"""
Random Forest Classifier was selected as the final model because it achieved
the highest accuracy among all tested algorithms.
"""
)