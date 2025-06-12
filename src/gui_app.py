# src/gui_app.py

import os
import streamlit as st
import pandas as pd
import pickle
from preprocess import preprocess_data, clean_data

# Ensure the data directory and model file exist
model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'student_model.pkl')
model_path = os.path.abspath(model_path)
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please train the model first.")
    st.stop()

with open(model_path, 'rb') as f:
    model = pickle.load(f)

st.title("📊 Student Mental Health Predictor")

st.write("Fill in the form below to get a prediction of your Academic Performance Change:")

gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.number_input("Age", min_value=10, max_value=100, value=20)
education_level = st.selectbox("Education Level", ["High School", "Undergraduate", "Postgraduate"])
screen_time = st.slider("Screen Time (hrs/day)", 0.0, 12.0, 3.0)
sleep_duration = st.slider("Sleep Duration (hrs)", 0.0, 12.0, 7.0)
physical_activity = st.slider("Physical Activity (hrs/week)", 0.0, 20.0, 3.0)
stress = st.selectbox("Stress Level", [1, 2, 3, 4, 5])
anxious = st.selectbox("Anxious Before Exams", ["Yes", "No"])

if st.button("Predict"):
    # Create input DataFrame
    input_df = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "Education Level": education_level,
        "Screen Time (hrs/day)": screen_time,
        "Sleep Duration (hrs)": sleep_duration,
        "Physical Activity (hrs/week)": physical_activity,
        "Stress Level": stress,
        "Anxious Before Exams": anxious
    }])

    input_cleaned = clean_data(input_df)
    X_input, _ = preprocess_data(input_cleaned, target="Academic Performance Change")

    # Align columns with the model's expected features
    expected_features = model.feature_names_in_
    for col in expected_features:
        if col not in X_input.columns:
            X_input[col] = 0  # or another default value
    X_input = X_input[expected_features]

    prediction = model.predict(X_input)[0]
    st.success(f"🎯 Predicted Academic Performance Change: {prediction}")
