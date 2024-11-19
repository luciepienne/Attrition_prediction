"""This module contains the main FastAPI application for employee attrition prediction.
It includes endpoints for authentication and prediction using various machine learning models."""

import os

import requests
import streamlit as st

API_URL = "http://localhost:8001"


def get_token(username: str, password: str) -> str:
    """Obtenir un token d'authentification."""
    try:
        response = requests.post(
            f"{API_URL}/token",
            data={"username": username, "password": password},
            timeout=10,
        )
        response.raise_for_status()
        return response.json().get("access_token")
    except requests.RequestException:
        st.error("Invalid credentials or server error")
        return None


def predict_employee_attrition(token: str, employee_data: dict) -> dict:
    """Faire une prédiction sur l'attrition des employés."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    try:
        response = requests.post(
            f"{API_URL}/predict", headers=headers, json=employee_data, timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        st.error("Error in prediction or server error")
        return None


def login_page():
    """Page de connexion."""
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        token = get_token(username, password)
        if token:
            st.session_state["token"] = token
            st.success("Logged in successfully!")
            st.session_state["page"] = "predict"


def get_employee_data():
    """Collecter les données de l'employé."""
    return {
        "Gender": st.selectbox("Gender", ["Male", "Female"]),
        "Age": st.number_input("Age", min_value=18, max_value=65),
        "MaritalStatus": st.selectbox(
            "Marital Status", ["Single", "Married", "Divorced"]
        ),
        "Education": st.selectbox("Education Level", [1, 2, 3, 4, 5]),
        "EducationField": st.selectbox(
            "Education Field",
            [
                "Human Resources",
                "Life Sciences",
                "Marketing",
                "Medical",
                "Other",
                "Technical Degree",
            ],
        ),
        "NumCompaniesWorked": st.number_input(
            "Number of Companies Worked", min_value=1
        ),
        "WorkExperience": st.number_input(
            "Total Years of Work Experience", min_value=0.0
        ),
        "Department": st.selectbox(
            "Department", ["Human Resources", "Research & Development", "Sales"]
        ),
        "JobRole": st.selectbox(
            "Job Role", ["Manager", "Sales Executive", "Healthcare Representative"]
        ),
        "JobLevel": st.number_input("Job Level (1-5)", min_value=1, max_value=5),
        "DistanceFromHome": st.number_input(
            "Distance from Home (miles)", min_value=0.0
        ),
        "BusinessTravel": st.selectbox(
            "Business Travel Frequency",
            ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
        ),
        "OverTime": st.selectbox("Works Overtime?", ["Yes", "No"]),
        "JobInvolvement": st.number_input(
            "Job Involvement Level (1-4)", min_value=1, max_value=4
        ),
        "PercentSalaryHike": st.number_input(
            "Percentage of Salary Hike (0-100)", min_value=0.0, max_value=100.0
        ),
        "MonthlyIncome": st.number_input("Monthly Income", min_value=0.0),
        "MonthlyRate": st.number_input("Monthly Rate", min_value=0),
        "DailyRate": st.number_input("Daily Rate", min_value=0),
        "HourlyRate": st.number_input("Hourly Rate", min_value=0),
        "StockOptionLevel": st.number_input(
            "Stock Option Level (0-3)", min_value=0, max_value=3
        ),
        "PerformanceRating": st.number_input(
            "Performance Rating (1-4)", min_value=1, max_value=4
        ),
        "TrainingTimesLastYear": st.number_input(
            "Training Times Last Year", min_value=0
        ),
        "YearsWithCurrManager": st.number_input(
            "Years with Current Manager", min_value=0.0
        ),
        "OverallSatisfaction": st.number_input(
            "Overall Job Satisfaction (1-5)", min_value=1.0, max_value=5.0
        ),
    }


def predict_page():
    """Page de prédiction."""
    st.title("Predict Employee Attrition")

    if "token" not in st.session_state:
        st.error("Please log in to access this page.")
        return

    employee_data = get_employee_data()

    if st.button("Predict"):
        predictions = predict_employee_attrition(
            st.session_state["token"], employee_data
        )

        if predictions and "best_model_name" in predictions:
            prediction_probability_percentage = predictions["prediction"] * 100
            risk_level = predictions["attrition_risk"]

            st.write(
                f"Model: {predictions['best_model_name']}, "
                f"Prediction Probability: {prediction_probability_percentage:.2f}%, "
                f"Risk: {risk_level}"
            )
        else:
            st.error("Error: Invalid response from the prediction API.")


if "page" not in st.session_state:
    st.session_state["page"] = "login"

if st.session_state["page"] == "login":
    login_page()
elif st.session_state["page"] == "predict":
    predict_page()
