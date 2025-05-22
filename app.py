import streamlit as st
import pickle
import numpy as np
import pandas as pd

def load_model():
    with open("maternity.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def load_label_encoder():
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return le

def main():
    st.title("Maatricare")
    st.subheader(" _Personalized_ _AI-Based_ :green[_Nutrition_] _&_ :green[_Maternal_ _Care_] ")
    st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

    option = st.sidebar.selectbox(
        "Choose a service",
        ("Check Symptoms (AI Assistant)", "Detect Signs of Malnutrition", "Personalized Diet Plan", "Tips for Newborn Care", "Maternity Risks"),
    )
    st.subheader(option)

    if option == "Maternity Risks":
        st.write("### Enter the following details:")

        age = st.number_input("Age", min_value=18, max_value=50, value=30)
        body_temp = st.number_input("Body Temperature (°F)", min_value=96.0, max_value=100.0, value=97.6)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=60, max_value=120, value=95)
        systolic_bp = st.number_input("Systolic BP (mm Hg)", min_value=90, max_value=170, value=120)
        diastolic_bp = st.number_input("Diastolic BP (mm Hg)", min_value=60, max_value=110, value=95)
        bmi = st.number_input("BMI", min_value=18.0, max_value=30.0, value=23.5)
        hba1c = st.number_input("HbA1c (%)", min_value=35.0, max_value=50.0, value=40.0)
        fasting_glucose = st.number_input("Fasting Glucose (mg/dL)", min_value=4.0, max_value=8.0, value=5.8)

        if st.button("Predict Maternity Risk"):
            model = load_model()
            le = load_label_encoder()
            
            input_df = pd.DataFrame(
                [[age, body_temp, heart_rate, systolic_bp, diastolic_bp, bmi, hba1c, fasting_glucose]],
                columns=['Age', 'BodyTemp', 'HeartRate', 'SystolicBP', 'DiastolicBP', 'BMI', 'HbA1c', 'FastingGlucose']
            )
            
            pred_encoded = model.predict(input_df)
            predicted_class = le.inverse_transform(pred_encoded)[0]
            
            st.success(f"Predicted risk level: {predicted_class}")

if __name__ == "__main__":
    main()
