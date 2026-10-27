
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# -------------------- PAGE CONFIGURATION --------------------

st.set_page_config(
    page_title="GradeWise | Student Analytics",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------- PROJECT PATHS --------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"

# These feature names must match the trained model.
FEATURES = ["studytime", "failures", "absences", "G1", "G2"]



# -------------------- CUSTOM CSS --------------------

st.markdown(
    """
    <style>
    /* Main application */
    .stApp {
        background-color: #F5F8FE;
        color: #172B4D;
    }

    [data-testid="stHeader"] {
        background-color: rgba(245, 248, 254, 0.95);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E5EAF3;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #24344D !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #52627A !important;
    }

    /* General headings and text */
    h1, h2, h3, h4 {
        color: #173B78;
    }

    p, li {
        color: #344563;
    }

    /* Form labels */
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label {
        color: #24344D !important;
    }

    /* Input fields */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
        background-color: #FFFFFF;
        border-color: #CBD5E1;
    }

    div[data-baseweb="select"] *,
    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea {
        color: #172B4D !important;
    }

    /* Slider and number input text */
    [data-testid="stSlider"] label,
    [data-testid="stNumberInput"] label {
        color: #24344D !important;
    }

    /* Hero banner */
    .hero {
        background: linear-gradient(120deg, #173B78, #3478E5);
        padding: 30px;
        border-radius: 18px;
        color: #FFFFFF;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: #FFFFFF !important;
        font-size: 34px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #EAF2FF !important;
        font-size: 16px;
        margin-bottom: 0;
    }

    /* Metric cards */
    .metric-card {
        background: #FFFFFF;
        padding: 22px;
        border: 1px solid #E5EAF3;
        border-radius: 15px;
        min-height: 130px;
        box-shadow: 0 4px 14px rgba(34, 67, 120, 0.04);
    }

    .metric-label {
        color: #52627A !important;
        font-size: 14px;
        margin-bottom: 10px;
    }

    .metric-value {
        color: #173B78 !important;
        font-size: 29px;
        font-weight: 750;
    }

    /* Section titles */
    .section-title {
        color: #173B78 !important;
        font-size: 22px;
        font-weight: 700;
        margin-top: 18px;
        margin-bottom: 10px;
    }

    /* Information cards */
    .info-card {
        background: #FFFFFF;
        border: 1px solid #E5EAF3;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 12px;
        color: #24344D;
    }

    .info-card p,
    .info-card span,
    .info-card li {
        color: #344563 !important;
    }

    /* Form container */
    [data-testid="stForm"] {
        background: #FFFFFF;
        border: 1px solid #E5EAF3;
        border-radius: 15px;
        padding: 22px;
    }

    /* Buttons */
    div.stButton > button[kind="primary"],
    div.stFormSubmitButton > button {
        background: #2563EB;
        color: #FFFFFF !important;
        border: none;
        border-radius: 9px;
        min-height: 45px;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"]:hover,
    div.stFormSubmitButton > button:hover {
        background: #1749B5;
        color: #FFFFFF !important;
    }

    /* Information and warning messages */
    [data-testid="stAlert"] p {
        color: #24344D !important;
    }

    /* Small notes */
    .small-note {
        color: #52627A !important;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)



# -------------------- LOAD TRAINED MODEL --------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained model was not found. "
            "Please run: python train.py"
        )

    model = joblib.load(MODEL_PATH)

    metadata = {}
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as file:
                metadata = json.load(file)
        except (json.JSONDecodeError, OSError):
            metadata = {}

    return model, metadata


try:
    model, metadata = load_model()
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.info("Check that models/model.joblib exists and dependencies are installed.")
    st.stop()


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.markdown("# 📘 GradeWise")
    st.caption("Student Performance Analytics")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "About Project"],
        index=0,
    )

    st.divider()

    st.markdown("### 🧠 Machine Learning")
    st.write("Model: Trained regression pipeline")
    st.write("Inputs: 5 student features")
    st.write("Output: Estimated final grade")

    st.divider()

    st.markdown("### 🛠️ Technologies")
    st.caption("Python · pandas · scikit-learn · Streamlit")

    st.markdown(
        '<p class="small-note">Educational project for learning '
        'and demonstrating machine learning.</p>',
        unsafe_allow_html=True,
    )


# -------------------- DASHBOARD PAGE --------------------

if page == "Dashboard":

    st.markdown(
        """
        <div class="hero">
            <h1>Student Performance Dashboard</h1>
            <p>Understand academic progress and estimate a student's
            final grade using a machine learning model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Enter Student Details</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Complete the form below. The model uses previous grades "
        "and study-related information to make an estimate."
    )

    # -------------------- INPUT FORM --------------------

    with st.form("student_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            studytime_label = st.selectbox(
                "Weekly Study Time",
                [
                    "Less than 2 hours",
                    "2 to 5 hours",
                    "5 to 10 hours",
                    "More than 10 hours",
                ],
                index=1,
                help="Select the student's approximate weekly study-time category.",
            )

            failures = st.number_input(
                "Past Class Failures",
                min_value=0,
                max_value=4,
                value=0,
                step=1,
                help="Number of previous class failures.",
            )

            absences = st.number_input(
                "Number of Absences",
                min_value=0,
                max_value=100,
                value=0,
                step=1,
                help="Number of recorded school absences.",
            )

        with col2:
            g1 = st.slider(
                "Earlier Grade (G1)",
                min_value=0,
                max_value=20,
                value=12,
                help="Earlier-period grade, on a 0–20 scale.",
            )

            g2 = st.slider(
                "Recent Grade (G2)",
                min_value=0,
                max_value=20,
                value=13,
                help="More recent grade, on a 0–20 scale.",
            )

            st.markdown(
                """
                <div class="info-card">
                    <strong>📌 Grade scale</strong><br>
                    <span class="small-note">
                    Grades range from 0 to 20, matching the dataset
                    used to train this model.
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        submitted = st.form_submit_button(
            "✨ Predict Final Grade",
            type="primary",
            use_container_width=True,
        )

    # Keep the selected inputs available after form submission.
    if submitted:
        studytime_map = {
            "Less than 2 hours": 1,
            "2 to 5 hours": 2,
            "5 to 10 hours": 3,
            "More than 10 hours": 4,
        }

        input_values = {
            "studytime": studytime_map[studytime_label],
            "failures": int(failures),
            "absences": int(absences),
            "G1": int(g1),
            "G2": int(g2),
        }

        input_df = pd.DataFrame([input_values], columns=FEATURES)

        try:
            prediction = float(model.predict(input_df)[0])
            prediction = max(0.0, min(20.0, prediction))

            st.session_state["gradewise_result"] = {
                "prediction": prediction,
                "inputs": input_values,
                "studytime_label": studytime_label,
            }

        except Exception as error:
            st.error(f"Prediction failed: {error}")

    # -------------------- RESULTS AND CHARTS --------------------

    result = st.session_state.get("gradewise_result")

    if result is not None:
        prediction = result["prediction"]
        inputs = result["inputs"]

        if prediction >= 16:
            category = "Higher estimate"
            category_color = "#15803D"
        elif prediction >= 10:
            category = "Middle estimate"
            category_color = "#B45309"
        else:
            category = "Lower estimate"
            category_color = "#B91C1C"

        previous_average = (inputs["G1"] + inputs["G2"]) / 2

        st.divider()

        st.markdown(
            '<div class="section-title">Prediction Results</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Results are generated by your saved machine learning model."
        )

        card1, card2, card3 = st.columns(3)

        with card1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Estimated Final Grade</div>
                    <div class="metric-value">{prediction:.2f}<span
                    style="font-size:16px;color:#64748B"> / 20</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with card2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Previous Grade Average</div>
                    <div class="metric-value">{previous_average:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with card3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Estimate Category</div>
                    <div class="metric-value"
                    style="font-size:22px;color:{category_color}">
                    {category}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="section-title">Grade Comparison</div>',
            unsafe_allow_html=True,
        )

        chart_data = pd.DataFrame(
            {
                "Grade": [
                    inputs["G1"],
                    inputs["G2"],
                    prediction,
                ]
            },
            index=[
                "Earlier Grade (G1)",
                "Recent Grade (G2)",
                "Predicted Final Grade",
            ],
        )

        chart_col, details_col = st.columns([3, 2])

        with chart_col:
            st.markdown("#### Academic Grade Overview")
            st.bar_chart(
                chart_data,
                y="Grade",
                color="#3478E5",
                height=300,
                y_label="Grade out of 20",
            )

        with details_col:
            st.markdown("#### Student Summary")

            st.markdown(
                f"""
                <div class="info-card">
                    <strong>Weekly study time</strong><br>
                    {result["studytime_label"]}
                    <hr>
                    <strong>Past failures</strong><br>
                    {inputs["failures"]}
                    <hr>
                    <strong>Absences</strong><br>
                    {inputs["absences"]}
                    <hr>
                    <strong>Predicted grade</strong><br>
                    {prediction:.2f} / 20
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.progress(
            min(1.0, max(0.0, prediction / 20)),
            text=f"Estimated grade level: {prediction:.1f} out of 20",
        )

        st.info(
            "This is an estimate based on historical data, not a guarantee "
            "of actual academic performance. It should not be used as the "
            "sole basis for important educational decisions."
        )

    else:
        st.markdown(
            """
            <div class="info-card">
                <strong>👋 Welcome to GradeWise</strong><br><br>
                Enter the student's details and select
                <strong>Predict Final Grade</strong> to view the estimated
                grade, summary cards, and comparison chart.
            </div>
            """,
            unsafe_allow_html=True,
        )


# -------------------- ABOUT PROJECT PAGE --------------------

else:
    st.markdown(
        """
        <div class="hero">
            <h1>About GradeWise</h1>
            <p>A student machine learning project for academic
            performance estimation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🎯 Project Objective")

    st.write(
        "GradeWise estimates a student's final grade using study time, "
        "past class failures, absences, and two earlier grades."
    )

    st.markdown("### ⚙️ How It Works")

    st.markdown(
        """
        1. Enter the student's information.
        2. The application prepares the five model input features.
        3. The saved machine learning model predicts the final grade.
        4. The dashboard displays the estimate and a grade comparison chart.
        """
    )

    st.markdown("### 🧰 Technology Stack")

    tech1, tech2, tech3 = st.columns(3)

    with tech1:
        st.markdown(
            '<div class="metric-card"><div class="metric-label">'
            'Programming</div><div class="metric-value">Python</div></div>',
            unsafe_allow_html=True,
        )

    with tech2:
        st.markdown(
            '<div class="metric-card"><div class="metric-label">'
            'Machine Learning</div><div class="metric-value">'
            'scikit-learn</div></div>',
            unsafe_allow_html=True,
        )

    with tech3:
        st.markdown(
            '<div class="metric-card"><div class="metric-label">'
            'Web Application</div><div class="metric-value">'
            'Streamlit</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### 📊 Model Inputs")

    feature_table = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Meaning": [
                "Weekly study-time category (1–4)",
                "Number of past class failures",
                "Number of absences",
                "Earlier grade (0–20)",
                "Recent grade (0–20)",
            ],
        }
    )

    st.dataframe(feature_table, hide_index=True, use_container_width=True)

    st.markdown("### ⚠️ Limitations")

    st.write(
        "The prediction depends on the training data and model quality. "
        "It may be inaccurate for individual students or different "
        "educational settings. Predictions should be used for demonstration "
        "and learning, not as a definitive judgment of a student."
    )

    st.markdown("### 📚 Dataset Attribution")

    st.markdown(
        "[UCI Student Performance Dataset]"
        "(https://archive.ics.uci.edu/dataset/320/student+performance)"
    )

    st.caption("GradeWise — Student Performance Predictor")
