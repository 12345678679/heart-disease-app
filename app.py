import joblib
import pandas as pd
import streamlit as st

st.title("Heart Disease Prediction System")
st.caption("AIC354 Assignment 2 - learning project, not medical advice")

model = joblib.load("heart_model.pkl")
sex_encoder = joblib.load("sex_encoder.pkl")
cp_encoder = joblib.load("cp_encoder.pkl")

age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", list(sex_encoder.classes_))
cp = st.selectbox("Chest pain type", list(cp_encoder.classes_))
thalach = st.number_input("Maximum heart rate achieved", min_value=60, max_value=220, value=150)

if st.button("Predict"):
    row = pd.DataFrame([[age, sex_encoder.transform([sex])[0],
                         cp_encoder.transform([cp])[0], thalach]],
                       columns=["age", "sex", "cp", "thalach"])
    pred = model.predict(row)[0]
    prob = model.predict_proba(row)[0][1]
    if pred == 1:
        st.error(f"Heart disease likely (probability {prob:.0%})")
    else:
        st.success(f"No heart disease likely (probability of disease {prob:.0%})")
