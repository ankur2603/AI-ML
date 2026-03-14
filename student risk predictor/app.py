import streamlit as st
import pickle
import numpy as np

# -----------------------------
# Load Trained Model
# -----------------------------
with open("student_risk_model.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Student Risk Predictor", page_icon="🎓", layout="wide")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📚 Student Risk Predictor")
st.sidebar.info("""
This app predicts if a student is at **academic risk** based on study habits, parental support, and extracurriculars.  
Built with **Python, Scikit-Learn, and Streamlit**.  
""")

# -----------------------------
# Main Title
# -----------------------------
st.markdown("<h1 style='text-align: center; color: #4B0082;'>🎓 Student Academic Risk Predictor</h1>", unsafe_allow_html=True)
st.write("Fill in the student details below to predict academic risk:")

# -----------------------------
# Input Section
# -----------------------------
with st.container():
    st.subheader("📝 Student Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=10, max_value=30, value=18)
        gender = st.selectbox("Gender", ["Male", "Female"])
        ethnicity = st.selectbox("Ethnicity", ["Group A", "Group B", "Group C", "Group D"])
        parental_education = st.selectbox(
            "Parental Education",
            ["Some High School", "High School", "Some College", "Bachelor", "Master"]
        )
        study_time = st.number_input("Study Time Weekly (hours)", min_value=0)
        absences = st.number_input("Number of Absences", min_value=0)
    
    with col2:
        tutoring = st.selectbox("Tutoring", ["No", "Yes"])
        parental_support = st.slider("Parental Support Level", 0, 4)
        extracurricular = st.selectbox("Extracurricular Participation", ["No", "Yes"])
        sports = st.selectbox("Sports Participation", ["No", "Yes"])
        music = st.selectbox("Music Participation", ["No", "Yes"])
        volunteering = st.selectbox("Volunteering Participation", ["No", "Yes"])

# -----------------------------
# Mapping
# -----------------------------
gender_map = {"Male": 0, "Female": 1}
ethnicity_map = {"Group A":0, "Group B":1, "Group C":2, "Group D":3}
education_map = {
    "Some High School":0, "High School":1, "Some College":2, "Bachelor":3, "Master":4
}
tutoring_map = {"No":0, "Yes":1}
extracurricular_map = {"No":0, "Yes":1}
sports_map = {"No":0, "Yes":1}
music_map = {"No":0, "Yes":1}
volunteering_map = {"No":0, "Yes":1}

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict Risk"):
    features = np.array([[
        age,
        gender_map[gender],
        ethnicity_map[ethnicity],
        education_map[parental_education],
        study_time,
        absences,
        tutoring_map[tutoring],
        parental_support,
        extracurricular_map[extracurricular],
        sports_map[sports],
        music_map[music],
        volunteering_map[volunteering]
    ]])
    
    prediction = model.predict(features)
    
    st.markdown("---")
    if prediction[0] == 1:
        st.markdown("<h2 style='color: red;'>⚠️ Student is at Academic Risk!</h2>", unsafe_allow_html=True)
        st.warning("Intervene early: Increase study support, reduce absences, encourage positive habits.")
    else:
        st.markdown("<h2 style='color: green;'>✅ Student is Safe</h2>", unsafe_allow_html=True)
        st.success("Keep up the good work! Encourage continued positive study habits.")
    
    st.balloons()  # Fun celebration if safe
