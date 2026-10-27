# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load models
models = {
    "KNN": joblib.load("knn_model.pkl"),
    "Decision Tree": joblib.load("decisiontree_model.pkl"),
    "Logistic Regression": joblib.load("logisticregression_model.pkl"),
    "Random Forest": joblib.load("randomforest_model.pkl"),
    "SVM": joblib.load("svm_model.pkl")
}

scaler = joblib.load("scaler.pkl")

st.title("🫀 Heart Stroke Prediction System")
st.sidebar.title("Model Selection")
model_choice = st.sidebar.selectbox("Choose a model", list(models.keys()))

# Input form
st.subheader("Enter Patient Details:")
age = st.number_input("Age", 1, 120, 50)
sex = st.selectbox("Sex", [0, 1])  # 0: Female, 1: Male
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Rest ECG", [0, 1, 2])
thalachh = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of ST", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1 = normal; 2 = fixed defect; 3 = reversable)", [1, 2, 3])

# Prediction
if st.button("Predict Stroke Risk"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalachh, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    prediction = models[model_choice].predict(input_scaled)[0]
    prob = models[model_choice].predict_proba(input_scaled)[0][1]

    st.success(f"Prediction: {'Stroke Risk' if prediction == 1 else 'No Stroke Risk'}")
    st.info(f"Probability: {prob * 100:.2f}%")
