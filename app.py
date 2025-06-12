import streamlit as st
import pickle
import pandas as pd
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Google API Key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API key is not set. Check your .env file or environment variables.")
else:
    genai.configure(api_key=api_key)

# Initialize Gemini Model
def get_gemini_response(prompt):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-002')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Load ML models
def load_model(filename):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    return model


def load_label_encoder(filename):
    with open(filename, "rb") as f:
        le = pickle.load(f)
    return le


st.set_page_config(page_title="Maatricare", layout='wide')


# Language Toggle
def main():
    language = st.sidebar.radio("Select Language / भाषा चुनें:", ("English", "हिन्दी"))

    if language == "English":
        st.title("Maatricare")
        st.subheader("_Personalized_ _AI-Based_ :green[_Nutrition_] _&_ :green[_Maternal_ _Care_]")
    else:
        st.title("मातृकेयर")
        st.subheader("_व्यक्तिगत_ _एआई आधारित_ :green[_पोषण_] _और_ :green[_मातृ_ _देखभाल_]")

    st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

    option = st.sidebar.selectbox(
        "Choose a service / सेवा चुनें:",
        (
            "Check Symptoms (AI Assistant) / लक्षण जांचें (एआई सहायक)",
            "Child Malnutrition / बाल कुपोषण",
            "Fetus Ultrasound Analyzer / भ्रूण अल्ट्रासाउंड विश्लेषक",
            "Tips for Newborn Care / नवजात देखभाल सुझाव",
            "Maternity Risks / मातृत्व जोखिम",
        ),
    )

    st.subheader(option)

    if "Maternity Risks" in option:
        if language == "English":
            st.subheader("Enter the following details:")
        else:
            st.subheader("निम्नलिखित विवरण दर्ज करें:")

        age = st.number_input("Age / उम्र", min_value=18, max_value=60, value=18)
        body_temp = st.number_input("Body Temperature (°F) / शरीर का तापमान (°F)", min_value=40.0, max_value=104.0, value=97.6)
        heart_rate = st.number_input("Heart Rate (bpm) / ह्रदय गति (बीपीएम)", min_value=45, max_value=150, value=72)
        systolic_bp = st.number_input("Systolic BP (mm Hg) / सिस्टोलिक बीपी (मिमी एचजी)", min_value=90, max_value=170, value=120)
        diastolic_bp = st.number_input("Diastolic BP (mm Hg) / डायस्टोलिक बीपी (मिमी एचजी)", min_value=40, max_value=140, value=80)
        bmi = st.number_input("BMI", min_value=15.0, max_value=30.0, value=21.0)
        hba1c = st.number_input("HbA1c (%)", min_value=30.0, max_value=50.0, value=40.0)
        fasting_glucose = st.number_input("Fasting Glucose (mg/dL) / उपवास ग्लूकोज (मिलीग्राम/डीएल)", min_value=3.0, max_value=9.0, value=5.8)

        if st.button("Predict Maternity Risk / मातृत्व जोखिम का पूर्वानुमान करें"):
            model = load_model("maternity.pkl")
            le = load_label_encoder("label_encoder.pkl")

            input_df = pd.DataFrame(
                [[age, body_temp, heart_rate, systolic_bp, diastolic_bp, bmi, hba1c, fasting_glucose]],
                columns=["Age", "BodyTemp", "HeartRate", "SystolicBP", "DiastolicBP", "BMI", "HbA1c", "FastingGlucose"],
            )

            pred_encoded = model.predict(input_df)
            predicted_class = le.inverse_transform(pred_encoded)[0]

            if language == "English":
                st.success(f"Predicted risk level: {predicted_class}")
            else:
                st.success(f"पूर्वानुमानित जोखिम स्तर: {predicted_class}")

    elif "Child Malnutrition" in option:
        if language == "English":
            st.subheader("Enter child details:")
        else:
            st.subheader("बच्चे का विवरण दर्ज करें:")

        columns = ['Sex', 'Age', 'Height', 'Weight', 'Low Income', 'Lower Middle Income', 'Upper Middle Income']

        sex = st.selectbox("Gender / लिंग", options=[0, 1], format_func=lambda x: "Female / महिला" if x == 0 else "Male / पुरुष")
        age = st.number_input("Age (years) / उम्र (वर्ष)", min_value=0, max_value=18, value=3)
        height = st.number_input("Height (cm) / ऊंचाई (सेमी)", min_value=30, max_value=200, value=88)
        weight = st.number_input("Weight (kg) / वजन (किग्रा)", min_value=1, max_value=150, value=13)

        income_level = st.selectbox(
            "Household Income Level / घरेलू आय स्तर",
            options=["Low Income", "Lower Middle Income", "Upper Middle Income"]
        )

        low_income = 1 if income_level == "Low Income" else 0
        lower_middle_income = 1 if income_level == "Lower Middle Income" else 0
        upper_middle_income = 1 if income_level == "Upper Middle Income" else 0

        if st.button("Predict Malnutrition Status / कुपोषण स्थिति का पूर्वानुमान करें"):
            svm_model_mn = load_model("svm_model_mn.pkl")
            le_mn = load_label_encoder("label_encoder_mn.pkl")

            sample_input_df = pd.DataFrame(
                [[sex, age, height, weight, low_income, lower_middle_income, upper_middle_income]],
                columns=columns
            )

            pred_class = svm_model_mn.predict(sample_input_df)
            pred_label = le_mn.inverse_transform(pred_class)

            if language == "English":
                st.success(f"Predicted Malnutrition Status: {pred_label[0]}")
            else:
                st.success(f"पूर्वानुमानित कुपोषण स्थिति: {pred_label[0]}")

    elif "Check Symptoms" in option:
        symptoms = [
            "Nausea and Vomiting", "Fatigue or Tiredness", "Frequent Urination", "Breast Tenderness and Swelling",
            "Food Cravings and Aversions", "Mood Swings", "Bloating and Gas", "Constipation", "Heartburn and Indigestion",
            "Headaches", "Mild Cramping or Spotting", "Back Pain", "Shortness of Breath", "Leg Cramps",
            "Increased Vaginal Discharge", "Dizziness or Fainting", "Swollen Feet and Ankles", "Stretch Marks",
            "Linea Nigra", "Nasal Congestion or Nosebleeds", "Insomnia or Trouble Sleeping", "Itchy Skin",
            "Pelvic Pressure", "Braxton Hicks Contractions", "Leaking Breasts"
        ]

        st.subheader("AI Maternity Symptom Checker / एआई मातृत्व लक्षण जांच")

        selected_symptoms = st.multiselect("Select Symptoms / लक्षण चुनें", symptoms)

        if st.button("Get AI Advice / एआई सलाह प्राप्त करें"):
            if not selected_symptoms:
                if language == "English":
                    st.warning("Please select at least one symptom.")
                else:
                    st.warning("कृपया कम से कम एक लक्षण चुनें।")
            else:
                prompt = f"Provide possible causes, concerns, and recommended care advice for these maternity symptoms: {', '.join(selected_symptoms)}"

                with st.spinner("Generating response using Gemini AI... / Gemini AI से उत्तर उत्पन्न हो रहा है..."):
                    response = get_gemini_response(prompt)

                    st.markdown("🤖 **AI Suggestion / एआई सुझाव:**")
                    st.write(response)

    elif "Tips for Newborn Care" in option:
        st.subheader("AI Tips for Newborn Care / नवजात देखभाल के लिए एआई सुझाव")

        if st.button("Generate AI Tips / एआई सुझाव उत्पन्न करें"):
            prompt = "Give 5 cute and helpful tips for newborn baby care."

            with st.spinner("Generating tips using Gemini AI... / Gemini AI से सुझाव उत्पन्न हो रहे हैं..."):
                response = get_gemini_response(prompt)

                tips = response.split("\n")
                clean_tips = [tip.strip() for tip in tips if tip.strip()][:5]

                if language == "English":
                    st.markdown("### 💡 Newborn Care Tips:")
                else:
                    st.markdown("### 💡 नवजात देखभाल सुझाव:")

                for tip in clean_tips:
                    st.markdown(f"✅ {tip}")

    else:
        url = "https://ultrasoundanalyzer.streamlit.app/"
        url2 = "https://github.com/sakshams23/Ultrasound_Analyzer/tree/main/Ultrasound%20reports%20sample"

        if language == "English":
            st.write("Ultrasound Analyzer is an additional feature. Access it [here](%s)." % url)
            st.markdown("You can get some sample reports for testing [here](%s)." % url2)
        else:
            st.write("अल्ट्रासाउंड विश्लेषक एक अतिरिक्त सुविधा है। इसे [यहां](%s) एक्सेस करें।" % url)
            st.markdown("आप परीक्षण के लिए कुछ नमूना रिपोर्ट्स [यहां](%s) प्राप्त कर सकते हैं।" % url2)


if __name__ == "__main__":
    main()
