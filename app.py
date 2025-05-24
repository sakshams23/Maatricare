import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Maatricare", layout='wide')

def load_model(filename):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    return model

def load_label_encoder(filename):
    with open(filename, "rb") as f:
        le = pickle.load(f)
    return le

def main():
    st.title("Maatricare")
    st.subheader(" _Personalized_ _AI-Based_ :green[_Nutrition_] _&_ :green[_Maternal_ _Care_] ")
    st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

    option = st.sidebar.selectbox(
        "Choose a service",
        (
            "Check Symptoms (AI Assistant)",
            "Child Malnutrition",
            "Personalized Diet Plan",
            "Tips for Newborn Care",
            "Maternity Risks",
        ),
    )
    st.subheader(option)

    if option == "Maternity Risks":
        st.subheader("Enter the following details:")

        age = st.number_input("Age", min_value=18, max_value=60, value=18)
        body_temp = st.number_input("Body Temperature (°F)", min_value=40.0, max_value=104.0, value=97.6)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=45, max_value=150, value=72)
        systolic_bp = st.number_input("Systolic BP (mm Hg)", min_value=90, max_value=170, value=120)
        diastolic_bp = st.number_input("Diastolic BP (mm Hg)", min_value=40, max_value=140, value=80)
        bmi = st.number_input("BMI", min_value=15.0, max_value=30.0, value=21.0)
        hba1c = st.number_input("HbA1c (%)", min_value=30.0, max_value=50.0, value=40.0)
        fasting_glucose = st.number_input("Fasting Glucose (mg/dL)", min_value=3.0, max_value=9.0, value=5.8)

        if st.button("Predict Maternity Risk"):
            model = load_model("maternity.pkl")
            le = load_label_encoder("label_encoder.pkl")

            input_df = pd.DataFrame(
                [[age, body_temp, heart_rate, systolic_bp, diastolic_bp, bmi, hba1c, fasting_glucose]],
                columns=[
                    "Age",
                    "BodyTemp",
                    "HeartRate",
                    "SystolicBP",
                    "DiastolicBP",
                    "BMI",
                    "HbA1c",
                    "FastingGlucose",
                ],
            )

            pred_encoded = model.predict(input_df)
            predicted_class = le.inverse_transform(pred_encoded)[0]

            st.success(f"Predicted risk level: {predicted_class}")

    elif option == "Child Malnutrition":
        st.subheader("Enter child details:")

        columns = ['Sex', 'Age', 'Height', 'Weight', 'Low Income', 'Lower Middle Income', 'Upper Middle Income']

        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        age = st.number_input("Age (years)", min_value=0, max_value=18, value=3)
        height = st.number_input("Height (cm)", min_value=30, max_value=200, value=88)
        weight = st.number_input("Weight (kg)", min_value=1, max_value=150, value=13)
        low_income = st.selectbox("Low Income", options=[0, 1])
        lower_middle_income = st.selectbox("Lower Middle Income", options=[0, 1])
        upper_middle_income = st.selectbox("Upper Middle Income", options=[0, 1])

        if st.button("Predict Malnutrition Status"):
            svm_model_mn = load_model("svm_model_mn.pkl")
            le_mn = load_label_encoder("label_encoder_mn.pkl")

            sample_input_df = pd.DataFrame(
                [[sex, age, height, weight, low_income, lower_middle_income, upper_middle_income]],
                columns=columns
            )

            pred_class = svm_model_mn.predict(sample_input_df)
            pred_label = le_mn.inverse_transform(pred_class)

            st.success(f"Predicted Malnutrition Status: {pred_label[0]}")

    else:
        st.info("This feature is under development.")

if __name__ == "__main__":
    main()
