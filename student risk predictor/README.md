# Student Academic Risk Predictor

## Overview

This project builds a **Machine Learning model to identify students who are at academic risk** based on behavioral and demographic factors such as study time, absences, parental support, and extracurricular activities.

The goal is to demonstrate a **complete machine learning workflow**, including data analysis, feature engineering, model training, evaluation, and model saving.

---

## Problem Statement

Early identification of academically at-risk students can help educators intervene before performance declines significantly.

This project predicts whether a student is **at risk (1)** or **not at risk (0)** using historical student data.

---

## Dataset Features

The dataset includes the following features:

* Age
* Gender
* Ethnicity
* Parental Education
* StudyTimeWeekly
* Absences
* Tutoring
* ParentalSupport
* Extracurricular
* Sports
* Music
* Volunteering
* GPA
* GradeClass

A new feature **Risk** is created based on GPA:

Risk = 1 → GPA < 2.5 (At Risk)
Risk = 0 → GPA ≥ 2.5 (Safe)

---

## Project Workflow

The project follows a standard machine learning pipeline:

1. Data Loading
2. Data Inspection (missing values check)
3. Exploratory Data Analysis (EDA)
4. Feature Engineering (Risk column creation)
5. Feature Selection
6. Train-Test Split
7. Model Training
8. Model Evaluation
9. Feature Importance Analysis
10. Model Saving

---

## Exploratory Data Analysis

Several visualizations were created to understand the dataset:

* GPA Distribution
* Study Time vs GPA
* Absences vs GPA
* Risk Distribution

These visualizations help identify patterns that influence student performance.

---

## Models Used

Two machine learning models were trained and compared:

* Logistic Regression
* Random Forest Classifier

Random Forest performed better and was selected as the final model.

---

## Model Evaluation

Model performance was evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report (Precision, Recall, F1-score)

A confusion matrix visualization was also generated to better understand prediction performance.

---

## Feature Importance

Feature importance was extracted from the Random Forest model to identify which factors most influence academic risk.

Key influencing factors include:

* Absences
* Study Time
* Parental Support

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## Project Structure

student-risk-predictor/
│
├── data/
│   └── Student_performance_data.csv
│
├── train_model.py
├── student_risk_model.pkl
├── requirements.txt
└── README.md

---

## How to Run the Project

1. Clone the repository

2. Install required libraries

pip install -r requirements.txt

3. Run the training script

python train_model.py

The script will:

* perform data analysis
* train the model
* evaluate performance
* save the trained model

---

## Future Improvements

* Build a web interface using Flask or Streamlit
* Deploy the model as an API
* Add more advanced ML models
* Improve feature engineering

---

## Author

ANKUR

