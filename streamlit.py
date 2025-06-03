import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model, scaler, and feature names
with open("heart_model.pkl", "rb") as f:
    model, scaler, feature_names = pickle.load(f)

st.title("Heart Failure Prediction App")

st.write("Enter patient data to predict heart disease risk:")

# Collect input
input_dict = {
    "Age": st.number_input("Age", 1, 120, 50),
    "Sex": st.selectbox("Sex", options=[0, 1]),  # 0 = Female, 1 = Male
    "ChestPainType": st.selectbox("Chest Pain Type", options=[0, 1, 2, 3]),
    "RestingBP": st.number_input("Resting Blood Pressure", 50, 200, 120),
    "Cholesterol": st.number_input("Cholesterol", 0, 600, 200),
    "FastingBS": st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1]),
    "RestingECG": st.selectbox("Resting ECG", options=[0, 1, 2]),
    "MaxHR": st.number_input("Maximum Heart Rate", 60, 250, 150),
    "ExerciseAngina": st.selectbox("Exercise Induced Angina", options=[0, 1]),
    "Oldpeak": st.number_input("Oldpeak", 0.0, 10.0, 1.0),
    "ST_Slope": st.selectbox("ST Slope", options=[0, 1, 2])
}

if st.button("Predict Heart Disease Risk"):
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_names)  # Match order and columns
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease! (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Risk of Heart Disease. (Probability: {probability:.2f})")
