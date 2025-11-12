
### Complete and Clean Code for `04_Streamlit_Deployment.py`

# Portfolio Project 3: Streamlit Deployment Application
# Objective: Load the trained model and deploy it as an interactive web application.

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- Configuration and Model Loading ---
MODEL_PATH = 'churn_predictor_rf.joblib'
try:
    # Load the Random Forest model saved from 03_Telco_Churn_Model.py
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error("Error: Model file 'churn_predictor_rf.joblib' not found. Please run the 03_Telco_Churn_Model.py script first to train and save the model.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Application Interface ---

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

st.title("📞 Telco Customer Churn Predictor")
st.markdown("Enter customer details to predict if they are likely to churn (leave the company).")

# --- Sidebar Inputs ---

st.sidebar.header("Customer Profile Input")

def user_input_features():
    """Collects user input via sidebar sliders/select boxes."""
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    
    # We use SeniorCitizen=0 or 1, and map the output for user clarity
    senior_citizen = st.sidebar.selectbox('Senior Citizen', (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')
    
    # These are the original features the model was trained on
    monthly_charges = st.sidebar.slider('Monthly Charges (USD)', 20.0, 120.0, 70.0)
    total_charges = st.sidebar.slider('Total Charges (USD)', 100.0, 5000.0, 1500.0)
    
    # Contract Type (will be one-hot encoded below)
    contract = st.sidebar.selectbox('Contract Type', ('Month-to-month', 'One year', 'Two year'))
    
    # Create the DataFrame matching the one-hot encoded columns used in training (03_Telco_Churn_Model.py)
    # The model expects columns for all possible encoded values, even if they are 0.
    data = {
        # Numerical/Binary features
        'SeniorCitizen': senior_citizen,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        
        # One-Hot Encoded Gender (Male is the column kept after dropping 'Female')
        'Gender_Male': 1 if gender == 'Male' else 0, 
        
        # One-Hot Encoded Contract (Month-to-month was the base case, so we need the other two)
        'Contract_One year': 1 if contract == 'One year' else 0, 
        'Contract_Two year': 1 if contract == 'Two year' else 0 
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- Main Page Display and Prediction ---

st.subheader('Selected Input Parameters')
st.dataframe(input_df)

# The 'Predict' Button
if st.button('Predict Churn Likelihood'):
    
    # 1. Prediction (returns 0 or 1)
    prediction = model.predict(input_df)
    # Prediction Probability (returns two probabilities [P(0), P(1)])
    prediction_proba = model.predict_proba(input_df)
    
    # Format the result for the user
    churn_status = 'YES (High Risk)' if prediction[0] == 1 else 'NO (Low Risk)'
    
    # 2. Display Results
    st.subheader('Prediction Result')
    
    if prediction[0] == 1:
        st.error(f"The model predicts **CHURN** ({churn_status})")
    else:
        st.success(f"The model predicts **NO CHURN** ({churn_status})")
        
    st.markdown("---")
    
    st.subheader('Prediction Probability')
    
    prob_no_churn = prediction_proba[0][0] * 100
    prob_churn = prediction_proba[0][1] * 100
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Probability of NO Churn", f"{prob_no_churn:.2f}%")
    with col2:
        st.metric("Probability of CHURN", f"{prob_churn:.2f}%")
        
    st.info("Congratulations! You have successfully completed Project 3 (Deployment Phase) for your portfolio.")
    

# --- Footer ---
st.markdown("---")
st.caption("AI Portfolio Project | Built with Streamlit and Scikit-learn")
