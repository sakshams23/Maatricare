import streamlit as st
import pickle
import pandas as pd
from transformers import pipeline

st.set_page_config(page_title="Maatricare", layout='wide')


def load_model(filename):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    return model


def load_label_encoder(filename):
    with open(filename, "rb") as f:
        le = pickle.load(f)
    return le


# Load Hugging Face model for AI responses
@st.cache_resource
def load_huggingface_model():
    return pipeline("text-generation", model="gpt2")  # Lightweight, works offline


def main():
    st.title("Maatricare")
    st.subheader("_Personalized_ _AI-Based_ :green[_Nutrition_] _&_ :green[_Maternal_ _Care_]")
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
                columns=["Age", "BodyTemp", "HeartRate", "SystolicBP", "DiastolicBP", "BMI", "HbA1c", "FastingGlucose"],
            )

            pred_encoded = model.predict(input_df)
            predicted_class = le.inverse_transform(pred_encoded)[0]

            st.success(f"Predicted risk level: {predicted_class}")

    elif option == "Child Malnutrition":
        st.subheader("Enter child details:")

        columns = ['Sex', 'Age', 'Height', 'Weight', 'Low Income', 'Lower Middle Income', 'Upper Middle Income']

        sex = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        age = st.number_input("Age (years)", min_value=0, max_value=18, value=3)
        height = st.number_input("Height (cm)", min_value=30, max_value=200, value=88)
        weight = st.number_input("Weight (kg)", min_value=1, max_value=150, value=13)

        income_level = st.selectbox(
            "Household Income Level",
            options=["Low Income", "Lower Middle Income", "Upper Middle Income"]
        )

        low_income = 1 if income_level == "Low Income" else 0
        lower_middle_income = 1 if income_level == "Lower Middle Income" else 0
        upper_middle_income = 1 if income_level == "Upper Middle Income" else 0

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

    elif option == "Check Symptoms (AI Assistant)":
        symptoms = [
            "Nausea and Vomiting", "Fatigue or Tiredness", "Frequent Urination", "Breast Tenderness and Swelling",
            "Food Cravings and Aversions", "Mood Swings", "Bloating and Gas", "Constipation", "Heartburn and Indigestion",
            "Headaches", "Mild Cramping or Spotting", "Back Pain", "Shortness of Breath", "Leg Cramps",
            "Increased Vaginal Discharge", "Dizziness or Fainting", "Swollen Feet and Ankles", "Stretch Marks",
            "Linea Nigra", "Nasal Congestion or Nosebleeds", "Insomnia or Trouble Sleeping", "Itchy Skin",
            "Pelvic Pressure", "Braxton Hicks Contractions", "Leaking Breasts"
        ]

        st.subheader("AI Maternity Symptom Checker")
        st.write("Select the symptoms you're experiencing:")

        selected_symptoms = st.multiselect("Symptoms", symptoms)

        if st.button("Get AI Advice"):
            if not selected_symptoms:
                st.warning("Please select at least one symptom.")
            else:
                prompt = f"What could be the possible causes, concerns, and recommended care advice for these symptoms: {', '.join(selected_symptoms)}?"

                generator = load_huggingface_model()

                with st.spinner("Generating response..."):
                    output = generator(prompt, max_length=50, do_sample=True, temperature=0.7)
                    st.markdown("🤖 AI Suggestion:")
                    st.write(output[0]['generated_text'])

    elif option == "Tips for Newborn Care":
        st.subheader("AI Tips for Newborn Care")
        st.write("Here are some AI-generated suggestions to help you care for your newborn with love 💖:")

        generator = load_huggingface_model()

        prompt = (
            "Give 5 cute and helpful tips for newborn baby care. "
        )

        if st.button("Generate AI Tips"):
            with st.spinner("Thinking like a loving nanny...💭"):
                output = generator(prompt, max_length=100, do_sample=True, temperature=0.8)
                result = output[0]['generated_text']

                tips = result.split("\n")
                clean_tips = [tip.strip() for tip in tips if tip.strip()][:5]

                st.markdown("### 💡 Newborn Care Tips:")
                for tip in clean_tips:
                    st.markdown(f"✅ {tip}")

    else:
        st.info("This feature is under development.")


if __name__ == "__main__":
    main()
