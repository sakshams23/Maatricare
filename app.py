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

# Text mapping function
def t(eng, hindi, language):
    return eng if language == "English" else hindi

def main():
    st.title("Maatricare")
    st.subheader("_Personalized_ _AI-Based_ :green[_Nutrition_] _&_ :green[_Maternal_ _Care_]")

    # Language Toggle
    language = st.sidebar.radio("Language / भाषा", ["English", "हिंदी"])

    st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

    option = st.sidebar.selectbox(
        t("Choose a service", "सेवा चुनें", language),
        (
            t("Check Symptoms (AI Assistant)", "लक्षण जांचें (एआई सहायक)", language),
            t("Child Malnutrition", "बाल कुपोषण", language),
            t("Fetus Ultrasound Analyzer", "भ्रूण अल्ट्रासाउंड विश्लेषक", language),
            t("Tips for Newborn Care", "नवजात शिशु देखभाल सुझाव", language),
            t("Maternity Risks", "गर्भावस्था जोखिम", language),
        )
    )
    st.subheader(option)

    if option == t("Maternity Risks", "गर्भावस्था जोखिम", language):
        st.subheader(t("Enter the following details:", "निम्न विवरण दर्ज करें:", language))

        age = st.number_input(t("Age", "आयु", language), min_value=18, max_value=60, value=18)
        body_temp = st.number_input(t("Body Temperature (°F)", "शरीर का तापमान (°F)", language), min_value=40.0, max_value=104.0, value=97.6)
        heart_rate = st.number_input(t("Heart Rate (bpm)", "हृदय गति (बीपीएम)", language), min_value=45, max_value=150, value=72)
        systolic_bp = st.number_input(t("Systolic BP (mm Hg)", "सिस्टोलिक बीपी (मिमी एचजी)", language), min_value=90, max_value=170, value=120)
        diastolic_bp = st.number_input(t("Diastolic BP (mm Hg)", "डायस्टोलिक बीपी (मिमी एचजी)", language), min_value=40, max_value=140, value=80)
        bmi = st.number_input(t("BMI", "बीएमआई", language), min_value=15.0, max_value=30.0, value=21.0)
        hba1c = st.number_input(t("HbA1c (%)", "एचबीए1सी (%)", language), min_value=30.0, max_value=50.0, value=40.0)
        fasting_glucose = st.number_input(t("Fasting Glucose (mg/dL)", "फास्टिंग ग्लूकोज (मिग्रा/डीएल)", language), min_value=3.0, max_value=9.0, value=5.8)

        if st.button(t("Predict Maternity Risk", "गर्भावस्था जोखिम पूर्वानुमानित करें", language)):
            model = load_model("maternity.pkl")
            le = load_label_encoder("label_encoder.pkl")

            input_df = pd.DataFrame(
                [[age, body_temp, heart_rate, systolic_bp, diastolic_bp, bmi, hba1c, fasting_glucose]],
                columns=["Age", "BodyTemp", "HeartRate", "SystolicBP", "DiastolicBP", "BMI", "HbA1c", "FastingGlucose"]
            )

            pred_encoded = model.predict(input_df)
            predicted_class = le.inverse_transform(pred_encoded)[0]

            st.success(f"{t('Predicted risk level', 'पूर्वानुमानित जोखिम स्तर', language)}: {predicted_class}")

    elif option == t("Child Malnutrition", "बाल कुपोषण", language):
        st.subheader(t("Enter child details:", "बच्चे का विवरण दर्ज करें:", language))

        columns = ['Sex', 'Age', 'Height', 'Weight', 'Low Income', 'Lower Middle Income', 'Upper Middle Income']

        sex = st.selectbox(t("Gender", "लिंग", language), options=[0, 1], format_func=lambda x: t("Female", "महिला", language) if x == 0 else t("Male", "पुरुष", language))
        age = st.number_input(t("Age (years)", "आयु (वर्ष)", language), min_value=0, max_value=18, value=3)
        height = st.number_input(t("Height (cm)", "ऊंचाई (सेमी)", language), min_value=30, max_value=200, value=88)
        weight = st.number_input(t("Weight (kg)", "वजन (किग्रा)", language), min_value=1, max_value=150, value=13)

        income_level = st.selectbox(
            t("Household Income Level", "परिवार की आय स्तर", language),
            options=["Low Income", "Lower Middle Income", "Upper Middle Income"]
        )

        low_income = 1 if income_level == "Low Income" else 0
        lower_middle_income = 1 if income_level == "Lower Middle Income" else 0
        upper_middle_income = 1 if income_level == "Upper Middle Income" else 0

        if st.button(t("Predict Malnutrition Status", "कुपोषण स्थिति पूर्वानुमानित करें", language)):
            svm_model_mn = load_model("svm_model_mn.pkl")
            le_mn = load_label_encoder("label_encoder_mn.pkl")

            sample_input_df = pd.DataFrame(
                [[sex, age, height, weight, low_income, lower_middle_income, upper_middle_income]],
                columns=columns
            )

            pred_class = svm_model_mn.predict(sample_input_df)
            pred_label = le_mn.inverse_transform(pred_class)

            st.success(f"{t('Predicted Malnutrition Status', 'पूर्वानुमानित कुपोषण स्थिति', language)}: {pred_label[0]}")

    elif option == t("Check Symptoms (AI Assistant)", "लक्षण जांचें (एआई सहायक)", language):
        symptoms = [
            t("Nausea and Vomiting", "मतली और उल्टी", language),
            t("Fatigue or Tiredness", "थकान", language),
            t("Frequent Urination", "बार-बार पेशाब आना", language),
            t("Breast Tenderness and Swelling", "स्तनों में कोमलता और सूजन", language),
            t("Food Cravings and Aversions", "खाने की लालसा और अरुचि", language),
            t("Mood Swings", "मूड में बदलाव", language),
            t("Bloating and Gas", "फुलाव और गैस", language),
            t("Constipation", "कब्ज", language),
            t("Heartburn and Indigestion", "सीने में जलन और अपच", language),
            t("Headaches", "सिरदर्द", language),
            t("Mild Cramping or Spotting", "हल्के ऐंठन या स्पॉटिंग", language),
            t("Back Pain", "पीठ दर्द", language),
            t("Shortness of Breath", "सांस की कमी", language),
            t("Leg Cramps", "पैरों में ऐंठन", language),
            t("Increased Vaginal Discharge", "योनि स्राव में वृद्धि", language),
            t("Dizziness or Fainting", "चक्कर आना या बेहोशी", language),
            t("Swollen Feet and Ankles", "सूजे हुए पैर और टखने", language),
            t("Stretch Marks", "खिंचाव के निशान", language),
            t("Linea Nigra", "लिनिया निग्रा", language),
            t("Nasal Congestion or Nosebleeds", "नाक बंद होना या नाक से खून आना", language),
            t("Insomnia or Trouble Sleeping", "अनिद्रा या सोने में परेशानी", language),
            t("Itchy Skin", "त्वचा में खुजली", language),
            t("Pelvic Pressure", "श्रोणि में दबाव", language),
            t("Braxton Hicks Contractions", "ब्रेक्सटन हिक्स संकुचन", language),
            t("Leaking Breasts", "स्तनों से रिसाव", language)
        ]

        st.subheader(t("AI Maternity Symptom Checker", "एआई मातृत्व लक्षण चेकर", language))
        st.write(t("Select the symptoms you are experiencing:", "वे लक्षण चुनें जो आप अनुभव कर रही हैं:", language))

        selected_symptoms = st.multiselect(t("Symptoms", "लक्षण", language), symptoms)

        if st.button(t("Get AI Advice", "एआई सलाह प्राप्त करें", language)):
            if not selected_symptoms:
                st.warning(t("Please select at least one symptom.", "कृपया कम से कम एक लक्षण चुनें।", language))
            else:
                if language == "English":
                    prompt = f"Provide possible causes, concerns, and recommended care advice for these maternity symptoms: {', '.join(selected_symptoms)}"
                else:
                    prompt = f"इन मातृत्व लक्षणों के संभावित कारण, चिंताएं और देखभाल की सिफारिशें हिंदी में बताएं: {', '.join(selected_symptoms)}"

                with st.spinner(t("Generating response using Gemini AI...", "Gemini AI से उत्तर प्राप्त किया जा रहा है...", language)):
                    response = get_gemini_response(prompt)
                    st.markdown("🤖 **" + t("AI Suggestion", "एआई सुझाव", language) + ":**")
                    st.write(response)

    elif option == t("Tips for Newborn Care", "नवजात शिशु देखभाल सुझाव", language):
        st.subheader(t("AI Tips for Newborn Care", "नवजात शिशु देखभाल के लिए एआई सुझाव", language))
        st.write(t("Here are some AI-generated suggestions to help you care for your newborn with love 💖:", "यहां आपके नवजात शिशु की देखभाल के लिए कुछ एआई सुझाव हैं:", language))

        if st.button(t("Generate AI Tips", "एआई सुझाव प्राप्त करें", language)):
            prompt = "Give 5 cute and helpful tips for newborn baby care." if language == "English" else "नवजात शिशु की देखभाल के लिए 5 प्यारे और सहायक सुझाव हिंदी में दें।"

            with st.spinner(t("Generating tips using Gemini AI...", "Gemini AI से सुझाव प्राप्त किए जा रहे हैं...", language)):
                response = get_gemini_response(prompt)

                tips = response.split("\n")
                clean_tips = [tip.strip() for tip in tips if tip.strip()][:5]

                st.markdown("### 💡 " + t("Newborn Care Tips:", "नवजात शिशु देखभाल सुझाव:", language))
                for tip in clean_tips:
                    st.markdown(f"✅ {tip}")

    else:
        url = "https://ultrasoundanalyzer.streamlit.app/"
        url2 = "https://github.com/sakshams23/Ultrasound_Analyzer/tree/main/Ultrasound%20reports%20sample"
        st.write(t("Ultrasound Analyzer is an additional feature. Access it", "अल्ट्रासाउंड विश्लेषक एक अतिरिक्त सुविधा है। इसे यहां एक्सेस करें", language) + f" [here]({url}).")
        st.markdown(t("You can get some sample reports for testing", "परीक्षण के लिए कुछ नमूना रिपोर्ट प्राप्त कर सकते हैं", language) + f" [here]({url2}).")

if __name__ == "__main__":
    main()
