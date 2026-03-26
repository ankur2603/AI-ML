import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# -----------------------------
# 1. LOAD DATA
# -----------------------------

columns = ["mpg", "cylinders", "displacement", "horsepower",
           "weight", "acceleration", "model_year", "origin", "car_name"]

rows = []

with open(r"data\FeulEff data.csv", "r") as file:
    for line in file:
        parts = line.strip().split()
        numeric_part = parts[:8]
        car_name = " ".join(parts[8:])
        rows.append(numeric_part + [car_name])

df = pd.DataFrame(rows, columns=columns)


# -----------------------------
# 2. CLEANING
# -----------------------------

df = df.apply(lambda x: x.str.replace('"', '', regex=False))
df.replace("?", pd.NA, inplace=True)

num_cols = ["mpg", "cylinders", "displacement", "horsepower",
            "weight", "acceleration", "model_year", "origin"]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.dropna(inplace=True)

print("Rows after cleaning:", len(df))


# -----------------------------
# 3. EDA
# -----------------------------

plt.figure()
sns.histplot(df["mpg"], kde=True)
plt.title("MPG Distribution")
plt.show()

plt.figure()
sns.scatterplot(x=df["weight"], y=df["mpg"])
plt.title("Weight vs MPG")
plt.show()

plt.figure()
sns.scatterplot(x=df["horsepower"], y=df["mpg"])
plt.title("Horsepower vs MPG")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# -----------------------------
# 4. MODEL PREP
# -----------------------------

df_model = df.drop("car_name", axis=1)

X = df_model.drop("mpg", axis=1)
y = df_model["mpg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))


# -----------------------------
# 5. EVALUATION FUNCTION
# -----------------------------

def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n{name} Performance:")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

    return mae, rmse, r2


# -----------------------------
# 6. FEATURE SCALING (for LR only)
# -----------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----------------------------
# 7. MODEL TRAINING
# -----------------------------

# Linear Regression (scaled)
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Decision Tree (no scaling needed)
dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)


# -----------------------------
# 8. MODEL EVALUATION
# -----------------------------

lr_metrics = evaluate_model(
    lr_model, X_test_scaled, y_test, "Linear Regression (Scaled)"
)

dt_metrics = evaluate_model(
    dt_model, X_test, y_test, "Decision Tree"
)


# -----------------------------
# 9. FEATURE IMPORTANCE
# -----------------------------

importance = pd.Series(dt_model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

print("\nFeature Importance:\n")
print(importance)

plt.figure()
sns.barplot(x=importance.values, y=importance.index)
plt.title("Feature Importance (Decision Tree)")
plt.show()