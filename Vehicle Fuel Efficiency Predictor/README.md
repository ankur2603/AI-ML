# 🚗 Vehicle Fuel Efficiency Predictor

## 📌 Overview
This project predicts a vehicle's fuel efficiency (MPG - Miles Per Gallon) using machine learning techniques based on engine and vehicle features.

The goal is to understand how factors like engine size, horsepower, and weight impact fuel consumption.

---

## 📊 Dataset
The dataset contains the following features:

- mpg (target variable)
- cylinders
- displacement
- horsepower
- weight
- acceleration
- model_year
- origin
- car_name

---

## ⚙️ Data Preprocessing
- Handled messy raw data (space + tab separated format)
- Removed unwanted quotes
- Converted data types to numeric
- Handled missing values (`?`)
- Dropped irrelevant column (`car_name`)

---

## 📈 Exploratory Data Analysis (EDA)
Performed visual analysis to understand relationships:

- MPG distribution
- Weight vs MPG
- Horsepower vs MPG
- Correlation heatmap

### 🔍 Key Observations:
- Weight and horsepower have a strong negative relationship with MPG
- Higher engine displacement leads to lower fuel efficiency
- Newer vehicles tend to be more fuel efficient

---

## 🤖 Models Used

### 1. Linear Regression
- Assumes linear relationships between features and target
- Performed best overall

### 2. Decision Tree Regressor
- Captures non-linear patterns
- Slightly lower performance compared to Linear Regression

---

## 📏 Model Evaluation

| Model                  | MAE  | RMSE | R² Score |
|-----------------------|------|------|----------|
| Linear Regression     | 2.42 | 3.27 | 0.79     |
| Decision Tree         | 2.30 | 3.30 | 0.78     |

---

## 📊 Feature Importance (Decision Tree)

Top features affecting MPG:
- Displacement (most important)
- Horsepower
- Model Year
- Weight

---

## 🧠 Key Learnings
- Real-world datasets often require heavy cleaning
- Simpler models can outperform complex ones when relationships are linear
- Feature importance helps understand model decisions
- Feature scaling does not always improve performance

---

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 🚀 How to Run

1. Clone the repository
2. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn