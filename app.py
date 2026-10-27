
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

# Project file locations
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"

# Page settings
st.set_page_config(
    page_title="GradeWise | Student Performance Predictor",
    page_icon="📘",
    layout="centered"
)

st.title("📘 GradeWise")
st.subheader("Student Performance Predictor")

st.write(
    "Estimate a student's final grade using previous grades "
    "and selected study-related information."
)

st.info(
    "This application provides an educational estimate based on "
    "historical data. It cannot guarantee actual grades or measure "
    "a student's ability."
)

# Check that the model files exist
if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error("Model files are missing. Please run python train.py first.")
    st.stop()

# Load saved model and evaluation results
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

# User inputs
st.markdown("### Enter student details")

studytime = st.selectbox(
    "Weekly study time",
    options=[1, 2, 3, 4],
    format_func=lambda value: {
        1: "Less than 2 hours",
        2: "2 to 5 hours",
        3: "5 to 10 hours",
        4: "More than 10 hours"
    }[value]
)

failures = st.number_input(
    "Past class failures",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

absences = st.number_input(
    "Number of absences",
    min_value=0,
    max_value=100,
    value=0,
    step=1
)

g1 = st.slider("Earlier grade (G1)", 0, 20, 10)
g2 = st.slider("Later grade (G2)", 0, 20, 10)

# Predict when the user clicks the button
if st.button("Predict final grade", type="primary"):
    student = pd.DataFrame([{
        "studytime": studytime,
        "failures": failures,
        "absences": absences,
        "G1": g1,
        "G2": g2
    }])

    prediction = float(model.predict(student)[0])
    prediction = max(0.0, min(20.0, prediction))

    st.markdown("### Prediction result")
    st.metric("Estimated final grade", f"{prediction:.1f} / 20")

    st.caption(
        "The actual grade may differ from this estimate."
    )

# Show model evaluation information
with st.expander("View model evaluation"):
    st.write(f"MAE: {metadata['mae']}")
    st.write(f"RMSE: {metadata['rmse']}")
    st.write(f"R² score: {metadata['r2']}")
    st.write(f"Test records: {metadata['test_rows']}")

    st.caption(
        "Evaluation metrics describe performance on the held-out test "
        "data; they do not guarantee accuracy for new students."
    )

st.divider()
st.caption("GradeWise | Machine Learning Project")
