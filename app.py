import streamlit as st




def main():
    st.title("Maatricare")
    st.subheader(" _Personalized_ _AI-Based_ :green[_Nutrition_] _&_ :green[_Maternal_ _Care_] ")
    st.markdown("<hr style='border:1px solid gray'>", unsafe_allow_html=True)

    
    option = st.selectbox(
    "Choose a service",
    ("Check Symptoms (AI Assistant)", "Detect Signs of Malnutrition", "Personalized Diet Plan", "Tips for Newborn Care"),
    )



 

if __name__=='__main__':
    main()
