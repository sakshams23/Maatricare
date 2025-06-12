import streamlit as st
import pickle
import pandas as pd
import os
from dotenv import load_dotenv
import google.generativeai as genai
from googletrans import Translator

# Load environment variables
load_dotenv()

# Configure Google API Key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API key is not set. Check your .env file or environment variables.")
else:
    genai.configure(api_key=api_key)

translator = Translator()

# Gemini Model
def get_gemini_response(prompt):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-002')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Model Loaders
def load_model(filename):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    return model

def load_label_encoder(filename):
    with open(filename, "rb") as f:
        le = pickle.load(f)
    return le

# Page Setup
st.set_page_config(page_title="Maatricare", layout='wide')

# Language Toggle
language = st.sidebar.radio("Select Language / भाषा चुनें", ["English", "हिन्दी"])

# Multilingual Labels
def t(english, hindi):
    return english if language == "English" else hindi

# App Title
st.title(t("Maatricare", "मातृकेयर"))
st.subheader(t("_Personalized_ _AI-Based_ :green[_Nutrition_] _&_ :green[_Maternal_ _Care_]", 
               "_व्यक्तिगत_ _AI-आधारित_ :green[_पोषण_] _&_ :green[_मातृत्व_ _देखभाल_]"))
st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

# Service Selector
option = st.sidebar.selectbox(
    t("Choose a service", "सेवा चुनें"),
    (
        t("Check Symptoms (AI Assistant)", "लक्षण जांचें (AI सहायक)"),
        t("Child Malnutrition", "बच्चों में कुपोषण"),
        t("Fetus Ultrasound Analyzer", "भ्रूण अल्ट्रासाउंड विश्लेषक"),
        t("Tips for Newborn Care", "नवजात देखभाल के सुझाव"),
        t("Maternity Risks", "मातृत्व जोखिम"),
    )
)
st.subheader(option)

if option == t("Maternity Risks", "मातृत्व जोखिम"):
    st.subheader(t("Enter the following details:", "निम्न विवरण दर्ज करें:"))

    age = st.number_input(t("Age", "आयु"), min_value=18, max_value=60, value=18)
    body_temp = st.number_input(t("Body Temperature (°F)", "शरीर का तापमान (°F)"), min_value=40.0, max_value=104.0, value=97.6)
    heart_rate = st.number_input(t("Heart Rate (bpm)", "हृदय गति (bpm)"), min_value=45, max_value=150, value=72)
    systolic_bp = st.number_input(t("Systolic BP (mm Hg)", "सिस्टोलिक बीपी (mm Hg)"), min_value=90, max_value=170, value=120)
    diastolic_bp = st.number_input(t("Diastolic BP (mm Hg)", "डायस्टोलिक बीपी (mm Hg)"), min_value=40, max_value=140, value=80)
    bmi = st.number_input("BMI", min_value=15.0, max_value=30.0, value=21.0)
    hba1c = st.number_input("HbA1c (%)", min_value=30.0, max_value=50.0, value=40.0)
    fasting_glucose = st.number_input(t("Fasting Glucose (mg/dL)", "उपवास ग्लूकोज (mg/dL)"), min_value=3.0, max_value=9.0, value=5.8)

    if st.button(t("Predict Maternity Risk", "मातृत्व जोखिम की भविष्यवाणी करें")):
        model = load_model("maternity.pkl")
        le = load_label_encoder("label_encoder.pkl")

        input_df = pd.DataFrame(
            [[age, body_temp, heart_rate, systolic_bp, diastolic_bp, bmi, hba1c, fasting_glucose]],
            columns=["Age", "BodyTemp", "HeartRate", "SystolicBP", "DiastolicBP", "BMI", "HbA1c", "FastingGlucose"],
        )

        pred_encoded = model.predict(input_df)
        predicted_class = le.inverse_transform(pred_encoded)[0]

        st.success(t(f"Predicted risk level: {predicted_class}", f"अनुमानित जोखिम स्तर: {predicted_class}"))

elif option == t("Child Malnutrition", "बच्चों में कुपोषण"):
    st.subheader(t("Enter child details:", "बच्चे का विवरण दर्ज करें:"))

    columns = ['Sex', 'Age', 'Height', 'Weight', 'Low Income', 'Lower Middle Income', 'Upper Middle Income']

    sex = st.selectbox(t("Gender", "लिंग"), options=[0, 1], format_func=lambda x: t("Female", "महिला") if x == 0 else t("Male", "पुरुष"))
    age = st.number_input(t("Age (years)", "आयु (वर्ष)"), min_value=0, max_value=18, value=3)
    height = st.number_input(t("Height (cm)", "लंबाई (सेमी)"), min_value=30, max_value=200, value=88)
    weight = st.number_input(t("Weight (kg)", "वजन (किग्रा)"), min_value=1, max_value=150, value=13)

    income_level = st.selectbox(
        t("Household Income Level", "परिवार की आय का स्तर"),
        options=[t("Low Income", "कम आय"), t("Lower Middle Income", "निम्न मध्यम आय"), t("Upper Middle Income", "उच्च मध्यम आय")]
    )

    low_income = 1 if income_level == t("Low Income", "कम आय") else 0
    lower_middle_income = 1 if income_level == t("Lower Middle Income", "निम्न मध्यम आय") else 0
    upper_middle_income = 1 if income_level == t("Upper Middle Income", "उच्च मध्यम आय") else 0

    if st.button(t("Predict Malnutrition Status", "कुपोषण स्थिति की भविष्यवाणी करें")):
        svm_model_mn = load_model("svm_model_mn.pkl")
        le_mn = load_label_encoder("label_encoder_mn.pkl")

        sample_input_df = pd.DataFrame(
            [[sex, age, height, weight, low_income, lower_middle_income, upper_middle_income]],
            columns=columns
        )

        pred_class = svm_model_mn.predict(sample_input_df)
        pred_label = le_mn.inverse_transform(pred_class)

        st.success(t(f"Predicted Malnutrition Status: {pred_label[0]}", f"अनुमानित कुपोषण स्थिति: {pred_label[0]}"))

elif option == t("Check Symptoms (AI Assistant)", "लक्षण जांचें (AI सहायक)"):
    symptoms_en = [
        "Nausea and Vomiting", "Fatigue or Tiredness", "Frequent Urination", "Breast Tenderness and Swelling",
        "Food Cravings and Aversions", "Mood Swings", "Bloating and Gas", "Constipation", "Heartburn and Indigestion",
        "Headaches", "Mild Cramping or Spotting", "Back Pain", "Shortness of Breath", "Leg Cramps",
        "Increased Vaginal Discharge", "Dizziness or Fainting", "Swollen Feet and Ankles", "Stretch Marks",
        "Linea Nigra", "Nasal Congestion or Nosebleeds", "Insomnia or Trouble Sleeping", "Itchy Skin",
        "Pelvic Pressure", "Braxton Hicks Contractions", "Leaking Breasts"
    ]

    symptoms_hi = [
        "मतली और उल्टी", "थकान या कमजोरी", "बार-बार पेशाब आना", "स्तनों में कोमलता और सूजन",
        "खाने की तीव्र इच्छा और अरुचि", "मूड स्विंग्स", "पेट फूलना और गैस", "कब्ज", "सीने में जलन और अपच",
        "सिरदर्द", "हल्की ऐंठन या धब्बे आना", "कमर दर्द", "सांस फूलना", "टांगों में ऐंठन",
        "योनि स्राव में वृद्धि", "चक्कर आना या बेहोशी", "पैर और टखनों में सूजन", "खिंचाव के निशान",
        "लिनिया निग्रा", "नाक बंद होना या नकसीर", "नींद की समस्या या अनिद्रा", "खुजली वाली त्वचा",
        "पेल्विक दबाव", "ब्रैक्सटन हिक्स संकुचन", "स्तनों से रिसाव"
    ]

    symptoms = symptoms_en if language == "English" else symptoms_hi

    st.subheader(t("AI Maternity Symptom Checker", "AI मातृत्व लक्षण जांचकर्ता"))
    st.write(t("Select the symptoms you're experiencing:", "जो लक्षण आप अनुभव कर रहे हैं उन्हें चुनें:"))

    selected_symptoms = st.multiselect(t("Symptoms", "लक्षण"), symptoms)

    if st.button(t("Get AI Advice", "AI सुझाव प्राप्त करें")):
        if not selected_symptoms:
            st.warning(t("Please select at least one symptom.", "कृपया कम से कम एक लक्षण चुनें।"))
        else:
            prompt = f"Provide possible causes, concerns, and recommended care advice for these maternity symptoms: {', '.join(selected_symptoms)}"

            with st.spinner(t("Generating response using Gemini AI...", "Gemini AI द्वारा प्रतिक्रिया उत्पन्न की जा रही है...")):
                response = get_gemini_response(prompt)

                if language == "हिन्दी":
                    response = translator.translate(response, dest='hi').text

                st.markdown(t("🤖 **AI Suggestion:**", "🤖 **AI सुझाव:**"))
                st.write(response)

elif option == t("Tips for Newborn Care", "नवजात देखभाल के सुझाव"):
    st.subheader(t("AI Tips for Newborn Care", "नवजात शिशु देखभाल के लिए AI सुझाव"))
    st.write(t("Here are some AI-generated suggestions to help you care for your newborn with love 💖:", 
               "अपने नवजात शिशु की देखभाल के लिए कुछ AI-सृजित सुझाव यहां दिए गए हैं 💖:"))

    if st.button(t("Generate AI Tips", "AI सुझाव उत्पन्न करें")):
        prompt = "Give 5 cute and helpful tips for newborn baby care."

        with st.spinner(t("Generating tips using Gemini AI...", "Gemini AI द्वारा सुझाव उत्पन्न किए जा रहे हैं...")):
            response = get_gemini_response(prompt)

            if language == "हिन्दी":
                response = translator.translate(response, dest='hi').text

            tips = response.split("\n")
            clean_tips = [tip.strip() for tip in tips if tip.strip()][:5]

            st.markdown(t("### 💡 Newborn Care Tips:", "### 💡 नवजात देखभाल सुझाव:"))
            for tip in clean_tips:
                st.markdown(f"✅ {tip}")

else:
    url = "https://ultrasoundanalyzer.streamlit.app/"
    url2 = "https://github.com/sakshams23/Ultrasound_Analyzer/tree/main/Ultrasound%20reports%20sample"
    st.write(t(f"Ultrasound Analyzer is an additional feature. Access it [here]({url}).",
               f"अल्ट्रासाउंड विश्लेषक एक अतिरिक्त सुविधा है। इसे [यहां]({url}) एक्सेस करें।"))
    st.markdown(t(f"You can get some sample reports for testing [here]({url2}).",
                  f"आप परीक्षण के लिए कुछ नमूना रिपोर्ट [यहां]({url2}) प्राप्त कर सकते हैं।"))
