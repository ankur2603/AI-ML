# 🎓 Student Academic Risk Predictor

**Predict which students are at academic risk using Machine Learning and a Streamlit web interface!**  

This project uses a **Random Forest classifier** to predict whether a student is at risk based on study habits, parental support, absences, and extracurricular activities. The project now includes a **fully interactive Streamlit app** for real-time predictions.

---

## 🧠 Motivation

Many students face challenges that affect their academic performance. Early identification of at-risk students can help educators and parents provide **targeted support**, improving outcomes and student well-being.  

This project demonstrates **practical use of AI & ML** in education and shows how to build a **user-friendly interface** for your ML model.

---

## ⚡ Features

- Input student details in **human-friendly dropdowns and sliders**  
- Predict if a student is **At Risk** or **Safe**  
- Color-coded predictions with emojis:  
  - ✅ Green: Safe  
  - ⚠️ Red: At Risk  
- Fun animation for safe predictions (Streamlit balloons)  
- Professional layout with columns and sidebar  

---

## 🛠 Technologies Used

- **Python 3.14**  
- **Pandas & NumPy** for data processing  
- **Scikit-Learn** for Machine Learning (Random Forest & Logistic Regression)  
- **Matplotlib & Seaborn** for Exploratory Data Analysis  
- **Streamlit** for interactive web app  
- **Pickle** to save/load trained model  

---

## 📊 Dataset

The dataset contains the following columns:

- Age, Gender, Ethnicity, Parental Education  
- Study Time Weekly, Absences  
- Tutoring, Parental Support, Extracurriculars  
- Sports, Music, Volunteering  
- GPA, Grade/Class  

**Target Variable:** `Risk` (1 = At Risk, 0 = Safe)  

*All categorical features are shown as human-readable labels in the Streamlit interface.*

---

