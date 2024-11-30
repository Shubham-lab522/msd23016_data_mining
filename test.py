import os
import pickle
import flask
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify
import pandas as pd
# Load the trained model
model_path = r"C:\Users\shubh\OneDrive\Desktop\project_dm\ML_Churn_Prediction\models\models\Logistic Regression_model.pkl"
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Define the columns used in training
columns = ['gender', 'SeniorCitizen', 
           'Partner', 'Dependents', 
           'tenure', 'PhoneService', 
           'MultipleLines', 'InternetService', 
           'OnlineSecurity', 'OnlineBackup', 
           'DeviceProtection', 'TechSupport', 
           'StreamingTV', 'StreamingMovies', 
           'Contract', 'PaperlessBilling', 
           'PaymentMethod', 'MonthlyCharges']

# Assuming you have used Label Encoding for categorical features
label_encoders = {
    'gender': LabelEncoder().fit(['Male', 'Female']),
    'Partner': LabelEncoder().fit(['Yes', 'No']),
    'Dependents': LabelEncoder().fit(['Yes', 'No']),
    'PhoneService': LabelEncoder().fit(['Yes', 'No']),
    'MultipleLines': LabelEncoder().fit(['Yes', 'No', 'No phone service']),
    'InternetService': LabelEncoder().fit(['DSL', 'Fiber optic', 'No']),
    'OnlineSecurity': LabelEncoder().fit(['Yes', 'No', 'No internet service']),
    'OnlineBackup': LabelEncoder().fit(['Yes', 'No', 'No internet service']),
    'DeviceProtection': LabelEncoder().fit(['Yes', 'No', 'No internet service']),
    'TechSupport': LabelEncoder().fit(['Yes', 'No', 'No internet service']),
    'StreamingTV': LabelEncoder().fit(['Yes', 'No', 'No internet service']),
    'StreamingMovies': LabelEncoder().fit(['Yes', 'No', 'No internet service']),
    'Contract': LabelEncoder().fit(['Month-to-month', 'One year', 'Two year']),
    'PaperlessBilling': LabelEncoder().fit(['Yes', 'No']),
    'PaymentMethod': LabelEncoder().fit(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
}

def main():
    st.title("Customer Churn Prediction")

    # Collect user input
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, value=0.0)
    # TotalCharges is removed as it is not used for training

    if st.button("Predict"):
        # Create a DataFrame from the input
        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [SeniorCitizen],
            'Partner': [Partner],
            'Dependents': [Dependents],
            'tenure': [tenure],
            'PhoneService': [PhoneService],
            'MultipleLines': [MultipleLines],
            'InternetService': [InternetService],
            'OnlineSecurity': [OnlineSecurity],
            'OnlineBackup': [OnlineBackup],
            'DeviceProtection': [DeviceProtection],
            'TechSupport': [TechSupport],
            'StreamingTV': [StreamingTV],
            'StreamingMovies': [StreamingMovies],
            'Contract': [Contract],
            'PaperlessBilling': [PaperlessBilling],
            'PaymentMethod': [PaymentMethod],
            'MonthlyCharges': [MonthlyCharges]
        })

        # Encode categorical features
        for col in label_encoders:
            if col in input_data:
                input_data[col] = label_encoders[col].transform(input_data[col])

        # Ensure the input data has the correct columns
        input_data = input_data.reindex(columns=columns, fill_value=0)

        # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)[0][1]

        st.write(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
        st.write(f"Prediction Probability: {prediction_proba:.2f}")

if __name__ == "__main__":
    main()