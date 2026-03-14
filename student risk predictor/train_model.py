import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("data/Student_performance_data _.csv")
print("Dataset loaded successfully")

print("\nMissing values:")
print(df.isnull().sum())



# -----------------------------
# Exploratory Data Analysis
# -----------------------------

#Plot GPA distribution to understand the overall academic performance of the students.
plt.figure(figsize=(8,5))
sns.histplot(df["GPA"], bins=20, kde=True)

plt.title("GPA Distribution")
plt.xlabel("GPA")
plt.ylabel("Number of Students")

plt.show()



#Students with higher study time tend to have slightly higher GPA.
plt.figure(figsize=(8,5))
sns.scatterplot(x=df["StudyTimeWeekly"], y=df["GPA"])

plt.title("Study Time vs GPA")
plt.xlabel("Study Time")
plt.ylabel("GPA")

plt.show()



#Students with more absences often show lower academic performance.
plt.figure(figsize=(8,5))
sns.scatterplot(x=df["Absences"], y=df["GPA"])

plt.title("Absences vs GPA")
plt.xlabel("Absences")
plt.ylabel("GPA")

plt.show()



# Create Risk column based on GPA
# Students with GPA < 2.5 are considered at risk

df["Risk"] = df["GPA"].apply(lambda x: 1 if x < 2.5 else 0)

print("\nRisk column created successfully")
print(df[["GPA", "Risk"]].head())

print("\nRisk distribution:")
print(df["Risk"].value_counts())


#Plot of the distribution of the Risk variable to see how many students are at risk vs safe. This helps us understand the balance of our dataset and the prevalence of at-risk students.
plt.figure(figsize=(6,4))
sns.countplot(x=df["Risk"])

plt.title("Distribution of Student Risk")
plt.xlabel("Risk (0 = Safe, 1 = At Risk)")
plt.ylabel("Number of Students")

plt.show()



# Drop columns that should not be used as features
df = df.drop(columns=["StudentID", "GPA", "GradeClass"])

print("Unnecessary columns removed.")
print("Remaining columns:")
print(df.columns)



# Separate features and target variable

X = df.drop(columns=["Risk"])
y = df["Risk"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)



# -----------------------------
# Train/Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)



# -----------------------------
# Model Training
# -----------------------------

# Logistic Regression model

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))



# Random Forest model

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

print("Model training completed")



# -----------------------------
# Model Evaluation
# -----------------------------

print("Confusion Matrix (Random Forest):")
print(confusion_matrix(y_test, y_pred_rf))


cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6,4))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")

plt.show()



print("Classification Report:")
print(classification_report(y_test, y_pred_rf))



# -----------------------------
# Feature Importance
# -----------------------------

feature_importance = rf_model.feature_importances_

features = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importance
})

features = features.sort_values(by="Importance", ascending=False)

print("Feature Importance:")
print(features)



plt.figure(figsize=(10,6))

plt.barh(features["Feature"], features["Importance"])

plt.xlabel("Importance")
plt.title("Feature Importance for Predicting Student Risk")

plt.gca().invert_yaxis()

plt.show()



# -----------------------------
# Save Model
# -----------------------------

with open("student_risk_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

print("Model saved successfully.")