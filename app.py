"""
app.py — Bank Customer Churn Predictor
Streamlit app by nmshah9 (Machine Learning App Developer)

Run locally with:
    streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras

# ----------------------------------------------------------------------
# Page config & custom branding (nmshah9)
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="nmshah9 | Bank Customer Churn Predictor",
    page_icon="🏦",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
BANNER_PATH = os.path.join(BASE_DIR, "banner.png")

# Custom CSS matching the nmshah9 navy / cyan brand palette used in the banner
st.markdown(
    """
    <style>
    .stApp { background-color: #0a0e27; }
    h1, h2, h3, h4, .stMarkdown, label, p, span, .stSelectbox label,
    .stSlider label, .stNumberInput label, .stRadio label {
        color: #e8ecf7 !important;
    }
    div[data-testid="stMetricValue"] { color: #4fd1e5 !important; }
    div[data-testid="stMetricLabel"] { color: #9aa8cc !important; }

    div[data-testid="stForm"] {
        background-color: #10173b;
        border: 1px solid #1e2a5e;
        border-radius: 16px;
        padding: 2rem;
    }

    .stButton>button, .stFormSubmitButton>button {
        background: linear-gradient(90deg, #2b5fd9 0%, #4fd1e5 100%);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        width: 100%;
    }
    .stButton>button:hover, .stFormSubmitButton>button:hover {
        background: linear-gradient(90deg, #4fd1e5 0%, #2b5fd9 100%);
        color: white;
    }

    .result-card {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 0.5rem;
    }
    .result-card .verdict { font-size: 1.4rem; font-weight: 800; }
    .result-card .pct { font-size: 2.6rem; font-weight: 800; margin: 0.3rem 0; }

    .footer-brand {
        text-align: center;
        color: #6b7a9e;
        font-size: 0.85rem;
        margin-top: 3rem;
        border-top: 1px solid #1e2a5e;
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Banner (custom nmshah9 branding image)
# ----------------------------------------------------------------------
if os.path.exists(BANNER_PATH):
    st.image(BANNER_PATH, width='stretch')

st.markdown(
    """
    <h1 style='text-align:center; margin-top: 1rem;'>🏦 Bank Customer Churn Predictor</h1>
    <p style='text-align:center; color:#9aa8cc; font-size:1.05rem;'>
        An AI-powered app by <b>nmshah9</b> — predicting customer churn with a
        Dropout-regularized ANN, tuned via KerasTuner.
    </p>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")


# ----------------------------------------------------------------------
# Load model artifacts (cached)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))

    models = {}
    baseline_path = os.path.join(MODELS_DIR, "baseline_final.keras")
    tuned_path = os.path.join(MODELS_DIR, "tuned_final.keras")
    if os.path.exists(baseline_path):
        models["Baseline ANN"] = keras.models.load_model(baseline_path)
    if os.path.exists(tuned_path):
        models["KerasTuner-Tuned ANN"] = keras.models.load_model(tuned_path)

    comparison_path = os.path.join(MODELS_DIR, "model_comparison.csv")
    comparison_df = pd.read_csv(comparison_path) if os.path.exists(comparison_path) else None

    return scaler, encoders, feature_columns, models, comparison_df


try:
    scaler, encoders, feature_columns, models, comparison_df = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model artifacts not found. Please run the notebook "
        "`Bank_Customer_Churn_Prediction.ipynb` (or `src/train_baseline.py` and "
        "`src/tune_model.py`) first to generate `models/*.pkl` and `models/*.keras`."
    )
    st.stop()


# ----------------------------------------------------------------------
# Feature engineering helper (must mirror src/preprocessing.py exactly)
# ----------------------------------------------------------------------
def build_feature_row(inputs: dict) -> pd.DataFrame:
    credit_score = inputs["CreditScore"]
    age = inputs["Age"]
    balance = inputs["Balance"]
    salary = inputs["EstimatedSalary"]
    tenure = inputs["Tenure"]

    balance_salary_ratio = balance / (salary + 1)
    is_zero_balance = int(balance == 0)
    tenure_by_age = tenure / (age + 1)

    bins = [0, 580, 670, 740, 800, 850]
    band = 1
    for i in range(len(bins) - 1):
        if bins[i] < credit_score <= bins[i + 1]:
            band = i + 1
            break
    if credit_score <= bins[0]:
        band = 1
    elif credit_score > bins[-1]:
        band = 5

    gender_encoded = encoders["gender_encoder"].transform([inputs["Gender"]])[0]

    row = {
        "CreditScore": credit_score,
        "Gender": gender_encoded,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": inputs["NumOfProducts"],
        "HasCrCard": inputs["HasCrCard"],
        "IsActiveMember": inputs["IsActiveMember"],
        "EstimatedSalary": salary,
        "BalanceSalaryRatio": balance_salary_ratio,
        "IsZeroBalance": is_zero_balance,
        "TenureByAge": tenure_by_age,
        "CreditScoreBand": band,
        "Geo_Germany": int(inputs["Geography"] == "Germany"),
        "Geo_Spain": int(inputs["Geography"] == "Spain"),
    }

    df_row = pd.DataFrame([row])
    df_row = df_row[feature_columns]  # enforce exact training column order
    return df_row


def run_prediction(inputs, model_name):
    X_row = build_feature_row(inputs)
    X_scaled = scaler.transform(X_row)
    model = models[model_name]
    prob = float(model.predict(X_scaled, verbose=0).ravel()[0])
    return prob


# ----------------------------------------------------------------------
# Customer input form (main page — no sidebar dependency)
# ----------------------------------------------------------------------
st.subheader("👤 Customer Profile")

with st.form("customer_form"):
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1:
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    with r1c2:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with r1c3:
        age = st.number_input("Age", min_value=18, max_value=92, value=38)
    with r1c4:
        credit_score = st.number_input("Credit Score", min_value=350, max_value=850, value=650)

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    with r2c1:
        tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=5)
    with r2c2:
        num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
    with r2c3:
        has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    with r2c4:
        is_active = st.selectbox("Active Member?", ["Yes", "No"])

    r3c1, r3c2, r3c3 = st.columns(3)
    with r3c1:
        balance = st.number_input("Account Balance ($)", min_value=0.0, max_value=300000.0, value=75000.0, step=1000.0)
    with r3c2:
        estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, max_value=300000.0, value=100000.0, step=1000.0)
    with r3c3:
        model_choice = st.selectbox("Model", list(models.keys()))

    submitted = st.form_submit_button("🔮 Predict Churn")

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Prediction result
# ----------------------------------------------------------------------
if submitted:
    inputs = {
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "CreditScore": credit_score,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_cr_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active == "Yes" else 0,
        "EstimatedSalary": estimated_salary,
    }

    prob = run_prediction(inputs, model_choice)
    will_churn = prob >= 0.5

    col1, col2 = st.columns([1, 1.3])

    with col1:
        if will_churn:
            st.markdown(
                f"<div class='result-card' style='background-color:#4a1621;'>"
                f"<div class='verdict' style='color:#ff6b81;'>⚠️ Likely to CHURN</div>"
                f"<div class='pct' style='color:#ff6b81;'>{prob*100:.1f}%</div>"
                f"<div style='color:#c98d97;'>estimated churn risk</div></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='result-card' style='background-color:#0f3d2e;'>"
                f"<div class='verdict' style='color:#4fd1a3;'>✅ Likely to STAY</div>"
                f"<div class='pct' style='color:#4fd1a3;'>{(1-prob)*100:.1f}%</div>"
                f"<div style='color:#8fbfae;'>estimated retention chance</div></div>",
                unsafe_allow_html=True,
            )
        st.caption(f"Model used: **{model_choice}**")

    with col2:
        st.markdown("##### Customer Snapshot")
        summary_df = pd.DataFrame({
            "Attribute": ["Geography", "Gender", "Age", "Credit Score", "Tenure",
                          "Balance", "Products", "Credit Card", "Active Member", "Salary"],
            "Value": [str(geography), str(gender), str(age), str(credit_score), f"{tenure} yrs",
                      f"${balance:,.0f}", str(num_products), str(has_cr_card), str(is_active), f"${estimated_salary:,.0f}"],
        })
        st.dataframe(summary_df, hide_index=True, width='stretch')
        st.progress(prob, text=f"Churn probability: {prob*100:.1f}%")
else:
    st.info("Fill in the customer profile above and click **🔮 Predict Churn**.")

st.markdown("---")

# ----------------------------------------------------------------------
# Model comparison table
# ----------------------------------------------------------------------
if comparison_df is not None:
    st.subheader("📊 Model Comparison (from training run)")
    st.dataframe(
        comparison_df.style.format({
            "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}",
            "F1-Score": "{:.4f}", "ROC-AUC": "{:.4f}",
        }).background_gradient(cmap="Blues", subset=["ROC-AUC"]),
        hide_index=True, width='stretch',
    )

st.markdown(
    """
    <div class='footer-brand'>
        Built with ❤️ by <b>nmshah9</b> · Machine Learning App Developer<br>
        AI-Powered · Data-Driven · Custom Solutions
    </div>
    """,
    unsafe_allow_html=True,
)
