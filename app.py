import streamlit as st
import pickle
import numpy as np

def load_model():
    with open("maternity.pkl", "rb") as f:
        model = pickle.load(f)
    return model

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

        age = st.number_input("Age", min_value=10, max_value=60, value=25)
        body_temp = st.number_input("Body Temperature (°F)", min_value=95.0, max_value=110.0, value=98.6)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=70)
        systolic_bp = st.number_input("Systolic BP (mm Hg)", min_value=80, max_value=200, value=120)
        diastolic_bp = st.number_input("Diastolic BP (mm Hg)", min_value=50, max_value=130, value=80)
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
        hba1c = st.number_input("HbA1c (%)", min_value=3.0, max_value=15.0, value=5.5)
        fasting_glucose = st.number_input("Fasting Glucose (mg/dL)", min_value=50, max_value=400, value=90)

        if st.button("Predict Maternity Risk"):
            model = load_model()
            input_data = np.array([[age, body_temp, heart_rate, systolic_bp, diastolic_bp, bmi, hba1c, fasting_glucose]])
            prediction = model.predict(input_data)

            st.success(f"Prediction: {prediction[0]}")

if __name__ == "__main__":
    main()
