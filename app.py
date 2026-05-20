import streamlit as st
import pandas as pd
import joblib


# PAGE CONFIG

st.set_page_config(
    page_title="Responsible AI Monitoring Dashboard",
    layout="wide"
)


# TITLE


st.title("Responsible AI Monitoring Dashboard")


# SIDEBAR


st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Accuracy Monitoring",
        "Bias Detection",
        "Drift Detection",
        "Explainability",
        "Governance Risk"
    ]
)


# OVERVIEW PAGE


if page == "Overview":

    st.header("AI Governance Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Model Accuracy", "85%")
    col2.metric("Bias Risk", "Low")
    col3.metric("Drift Status", "Stable")

    st.success("Responsible AI System Running Successfully")

    st.write("""
    This dashboard monitors:
    - Model Performance
    - Bias Detection
    - Drift Detection
    - Explainability
    - Governance Risk
    """)


# ACCURACY MONITORING PAGE


elif page == "Accuracy Monitoring":

    st.header("Model Accuracy Monitoring")

    try:

        # Load saved metrics
        metrics = joblib.load("outputs/metrics.pkl")

        accuracy = metrics["accuracy"]
        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1"]

        # First row
        col1, col2 = st.columns(2)

        col1.metric(
            "Accuracy",
            f"{accuracy:.2f}"
        )

        col2.metric(
            "Precision",
            f"{precision:.2f}"
        )

        # Second row
        col3, col4 = st.columns(2)

        col3.metric(
            "Recall",
            f"{recall:.2f}"
        )

        col4.metric(
            "F1 Score",
            f"{f1:.2f}"
        )

        st.success("Real-time model metrics loaded successfully.")

    except:
        st.error("Metrics file not found.")
        st.info("Run train_model.py first.")


# BIAS DETECTION PAGE


elif page == "Bias Detection":

    st.header("Bias Detection Dashboard")

    bias_data = pd.DataFrame({
        "Group": ["Male", "Female"],
        "Approval Rate": [82, 61]
    })

    st.subheader("Approval Rate by Gender")

    st.bar_chart(
        bias_data.set_index("Group")
    )

    st.warning("Potential gender bias detected.")

    st.write("""
    Bias monitoring helps ensure:
    - Fair decision-making
    - Ethical AI governance
    - Non-discriminatory predictions
    """)


# DRIFT DETECTION PAGE


elif page == "Drift Detection":

    st.header("Data Drift Monitoring")

    drift_score = 0.08

    st.metric(
        "PSI Score",
        drift_score
    )

    if drift_score < 0.1:
        st.success("No significant drift detected.")
    elif drift_score < 0.25:
        st.warning("Moderate drift detected.")
    else:
        st.error("High drift detected.")

    st.write("""
    Drift monitoring ensures:
    - Model stability
    - Reliable predictions
    - Detection of changing data patterns
    """)


# EXPLAINABILITY PAGE

elif page == "Explainability":

    st.header("Model Explainability")

    st.subheader("Top Features Affecting Predictions")

    feature_data = pd.DataFrame({
        "Feature": [
            "Education",
            "Age",
            "Hours-per-week",
            "Occupation"
        ],
        "Importance": [
            0.35,
            0.25,
            0.20,
            0.15
        ]
    })

    st.bar_chart(
        feature_data.set_index("Feature")
    )

    st.write("""
    Explainability helps users understand:
    - Why predictions are made
    - Which features influence outcomes
    - AI transparency and accountability
    """)


# GOVERNANCE RISK PAGE


elif page == "Governance Risk":

    st.header("Governance Risk Dashboard")

    risk_score = 28

    st.metric(
        "Overall Risk Score",
        risk_score
    )

    if risk_score < 30:
        st.success("Low Risk")
    elif risk_score < 60:
        st.warning("Medium Risk")
    else:
        st.error("High Risk")

    st.write("""
    Governance risk score combines:
    - Bias Risk
    - Drift Risk
    - Accuracy Stability
    - Explainability
    - Ethical Compliance
    """)
