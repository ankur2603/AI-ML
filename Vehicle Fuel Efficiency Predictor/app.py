import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Fuel Efficiency", layout="centered")

# -----------------------------
# GLASS + PURPLE PREMIUM CSS
# -----------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 900px;
}

/* Dark Gradient Background */
html, body, [class*="css"] {
    background: linear-gradient(135deg, #0f0c29, #1E1B2E, #2A2438);
}

/* Title */
h1 {
    text-align: center;
    font-weight: 600;
    color: #C4B5FD;
}

.subtitle {
    text-align: center;
    color: #A78BFA;
    margin-bottom: 2rem;
}

/* Glass Card */
.metric-box {
    text-align: center;
    padding: 1.8rem;
    border-radius: 16px;
   
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(14px);

    border: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.4);

    margin-bottom: 1.5rem;
}

/* Gradient text */
.metric-value {
    font-size: 2.8rem;
    font-weight: 600;
    background: linear-gradient(135deg, #A78BFA, #C4B5FD);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-label {
    font-size: 0.95rem;
    color: #C4B5FD;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #7C3AED, #A78BFA);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.5rem 1.2rem;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #6D28D9, #7C3AED);
}

/* Divider */
hr {
    border: none;
    height: 1px;
    background: #4C1D95;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# DARK GRAPH STYLE
# -----------------------------
plt.rcParams.update({
    "figure.facecolor": "#1E1B2E",
    "axes.facecolor": "#1E1B2E",
    "axes.edgecolor": "#444",
    "axes.labelcolor": "#EDE9FE",
    "xtick.color": "#C4B5FD",
    "ytick.color": "#C4B5FD",
    "text.color": "#EDE9FE"
})

sns.set_style("dark")

# -----------------------------
# TITLE
# -----------------------------
st.title("Fuel Efficiency")
st.markdown('<div class="subtitle">Estimate vehicle MPG with precision</div>', unsafe_allow_html=True)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
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

    df = df.apply(lambda x: x.str.replace('"', '', regex=False))
    df.replace("?", pd.NA, inplace=True)

    num_cols = ["mpg", "cylinders", "displacement", "horsepower",
                "weight", "acceleration", "model_year", "origin"]

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)
    return df


@st.cache_data
def train_model(df):
    df_model = df.drop("car_name", axis=1)

    X = df_model.drop("mpg", axis=1)
    y = df_model["mpg"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    return model, scaler


df = load_data()
model, scaler = train_model(df)

# -----------------------------
# INPUTS
# -----------------------------
st.subheader("Inputs")

col1, col2 = st.columns(2)

with col1:
    cylinders = st.slider("Cylinders", 3, 12, 4)
    displacement = st.number_input("Displacement", value=150.0)
    horsepower = st.number_input("Horsepower", value=100.0)

with col2:
    weight = st.number_input("Weight", value=2500.0)
    acceleration = st.number_input("Acceleration", value=15.0)
    model_year = st.slider("Model Year", 70, 82, 76)
    origin = st.selectbox("Origin", [1, 2, 3])

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict"):
    input_data = np.array([[cylinders, displacement, horsepower,
                            weight, acceleration, model_year, origin]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction < 18:
        label = "Low Efficiency"
    elif prediction < 28:
        label = "Moderate Efficiency"
    else:
        label = "High Efficiency"

    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value">{prediction:.2f} MPG</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# VISUALS
# -----------------------------
st.markdown("---")
st.subheader("Insights")

col3, col4 = st.columns(2)

with col3:
    fig1, ax1 = plt.subplots(dpi=140)
    sns.regplot(
        x=df["weight"], y=df["mpg"],
        ci=None,
        scatter_kws={"s":12, "alpha":0.6},
        line_kws={"color":"red", "linewidth":2.5},
        ax=ax1
    )
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.set_title("Weight Impact", fontsize=10)
    ax1.grid(False)
    st.pyplot(fig1)

with col4:
    fig2, ax2 = plt.subplots(dpi=140)
    sns.regplot(
        x=df["horsepower"], y=df["mpg"],
        ci=None,
        scatter_kws={"s":12, "alpha":0.6},
        line_kws={"color":"red", "linewidth":2.5},
        ax=ax2
    )
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.set_title("Horsepower Impact", fontsize=10)
    ax2.grid(False)
    st.pyplot(fig2)

# -----------------------------
# MODEL PERFORMANCE
# -----------------------------
st.markdown("---")
st.subheader("Model Reliability")

df_model = df.drop("car_name", axis=1)
X = df_model.drop("mpg", axis=1)
y = df_model["mpg"]

X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)

fig3, ax3 = plt.subplots(dpi=140)

sns.scatterplot(x=y, y=y_pred, s=15, ax=ax3)

ax3.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    linestyle="--",
    linewidth=1.5,
    color="red"
)

for spine in ax3.spines.values():
    spine.set_visible(False)

ax3.set_xlabel("Actual")
ax3.set_ylabel("Predicted")
ax3.set_title("Prediction Accuracy", fontsize=10)
ax3.grid(False)

st.pyplot(fig3)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Minimal • Intelligent • Premium UI")

