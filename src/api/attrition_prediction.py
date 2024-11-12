import requests
import streamlit as st

API_URL = "http://localhost:8001"


def get_token(username: str, password: str) -> str:
    """Obtenir un token d'authentification."""
    response = requests.post(
        f"{API_URL}/token", data={"username": username, "password": password}
    )
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        st.error("Invalid credentials")
        return None


def predict_employee_attrition(token: str, employee_data: dict) -> dict:
    """Faire une prédiction sur l'attrition des employés."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    response = requests.post(f"{API_URL}/predict", headers=headers, json=employee_data)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Error in prediction")
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
            # Rediriger vers la page de prédiction
            st.session_state["page"] = "predict"


def predict_page():
    """Page de prédiction."""
    st.title("Predict Employee Attrition")

    if "token" not in st.session_state:
        st.error("Please log in to access this page.")
        return

    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=65)
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    education = st.selectbox(
        "Education Level", [1, 2, 3, 4, 5]
    )  # Assuming these are the levels
    education_field = st.selectbox(
        "Education Field",
        [
            "Human Resources",
            "Life Sciences",
            "Marketing",
            "Medical",
            "Other",
            "Technical Degree",
        ],
    )
    num_companies_worked = st.number_input("Number of Companies Worked", min_value=0)
    work_experience = st.number_input("Total Years of Work Experience", min_value=0.0)
    department = st.selectbox(
        "Department", ["Human Resources", "Research & Development", "Sales"]
    )
    job_role = st.selectbox(
        "Job Role", ["Manager", "Sales Executive", "Healthcare Representative"]
    )
    job_level = st.number_input("Job Level (1-5)", min_value=1, max_value=5)
    distance_from_home = st.number_input("Distance from Home (miles)", min_value=0.0)
    business_travel = st.selectbox(
        "Business Travel Frequency",
        ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
    )
    over_time = st.selectbox("Works Overtime?", ["Yes", "No"])
    job_involvement = st.number_input(
        "Job Involvement Level (1-4)", min_value=1, max_value=4
    )
    percent_salary_hike = st.number_input(
        "Percentage of Salary Hike (0-100)", min_value=0.0, max_value=100.0
    )
    monthly_income = st.number_input("Monthly Income", min_value=0.0)
    monthly_rate = st.number_input("Monthly Rate", min_value=0)
    daily_rate = st.number_input("Daily Rate", min_value=0)
    hourly_rate = st.number_input("Hourly Rate", min_value=0)
    stock_option_level = st.number_input(
        "Stock Option Level (0-3)", min_value=0, max_value=3
    )
    performance_rating = st.number_input(
        "Performance Rating (1-4)", min_value=1, max_value=4
    )
    training_times_last_year = st.number_input("Training Times Last Year", min_value=0)
    years_with_curr_manager = st.number_input(
        "Years with Current Manager", min_value=0.0
    )
    overall_satisfaction = st.number_input(
        "Overall Job Satisfaction (1-5)", min_value=1.0, max_value=5.0
    )

    if st.button("Predict"):
        employee_data = {
            "Gender": gender,
            "Age": age,
            "MaritalStatus": marital_status,
            "Education": education,
            "EducationField": education_field,
            "NumCompaniesWorked": num_companies_worked,
            "WorkExperience": work_experience,
            "Department": department,
            "JobRole": job_role,
            "JobLevel": job_level,
            "DistanceFromHome": distance_from_home,
            "BusinessTravel": business_travel,
            "OverTime": over_time,
            "JobInvolvement": job_involvement,
            "PercentSalaryHike": percent_salary_hike,
            "MonthlyIncome": monthly_income,
            "MonthlyRate": monthly_rate,
            "DailyRate": daily_rate,
            "HourlyRate": hourly_rate,
            "StockOptionLevel": stock_option_level,
            "PerformanceRating": performance_rating,
            "TrainingTimesLastYear": training_times_last_year,
            "YearsWithCurrManager": years_with_curr_manager,
            "OverallSatisfaction": overall_satisfaction,
        }

        # Appel à l'API pour obtenir les prédictions
        predictions = predict_employee_attrition(st.session_state["token"], employee_data)

        # Vérifier si le format de la réponse est correct
        if predictions and 'best_model_name' in predictions:
            prediction_probability_percentage = predictions["prediction"] * 100
            risk_level = predictions["attrition_risk"]
            
            # Afficher les résultats
            st.write(f"Model: {predictions['best_model_name']}, Prediction Probability: {prediction_probability_percentage:.2f}%, Risk: {risk_level}")
        else:
            # Gérer le cas où la réponse ne contient pas les données attendues
            st.error("Error: Invalid response from the prediction API.")

if "page" not in st.session_state:
    st.session_state["page"] = "login"  # Page par défaut

if st.session_state["page"] == "login":
    login_page()
elif st.session_state["page"] == "predict":
    predict_page()


# # Interface principale
# st.sidebar.title("Navigation")
# page_options = ["Login", "Predict"]
# selected_page = st.sidebar.radio("Go to:", page_options)

# if selected_page == "Login":
#     login_page()
# elif selected_page == "Predict":
#     predict_page()
