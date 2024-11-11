"""Test functions for API and all its endpoints"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

# Ajouter le chemin du module parent pour l'importation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.app import app  # Assurez-vous que ce chemin est correct

client = TestClient(app)


def test_token_valid_credentials():
    """Test for valid user credentials."""
    response = client.post("/token", data={"username": "ADMIN", "password": "admin"})
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert "token_type" in response.json()


def test_token_invalid_credentials():
    """Test for invalid user credentials."""
    response = client.post("/token", data={"username": "wrong", "password": "wrong"})
    assert response.status_code == 400
    assert "detail" in response.json()


def test_predict_unauthorized():
    """Test prediction endpoint without authentication."""
    response = client.post(
        "/predict",
        params={
            "Gender": "Male",
            "Age": 30,
            "MaritalStatus": "Single",
            "Education": 3,
            "EducationField": "Life Sciences",
            "NumCompaniesWorked": 2,
            "WorkExperience": 5,
            "Department": "Sales",
            "JobRole": "Sales Executive",
            "JobLevel": 2,
            "DistanceFromHome": 10,
            "BusinessTravel": "Travel_Rarely",
            "OverTime": "Yes",
            "JobInvolvement": 3,
            "PercentSalaryHike": 15,
            "MonthlyIncome": 5000,
            "MonthlyRate": 10000,
            "DailyRate": 500,
            "HourlyRate": 50,
            "StockOptionLevel": 1,
            "PerformanceRating": 3,
            "TrainingTimesLastYear": 2,
            "YearsWithCurrManager": 3,
            "OverallSatisfaction": 4,
        },
    )
    assert response.status_code == 401


def test_predict_authorized():
    """Test prediction endpoint with valid authentication."""
    token_response = client.post(
        "/token", data={"username": "ADMIN", "password": "admin"}
    )
    token = token_response.json()["access_token"]

    response = client.post(
        "/predict",
        headers={"Authorization": f"Bearer {token}"},
        params={
            "Gender": "Female",
            "Age": 32,
            "MaritalStatus": "Divorced",
            "Education": 3,
            "EducationField": "Human Resources",
            "NumCompaniesWorked": 4,
            "WorkExperience": 10,
            "Department": "Human Resources",
            "JobRole": "Manager",
            "JobLevel": 3,
            "DistanceFromHome": 10,
            "BusinessTravel": "Non-Travel",
            "OverTime": "No",
            "JobInvolvement": 2,
            "PercentSalaryHike": 5,
            "MonthlyIncome": 5000,
            "MonthlyRate": 20,
            "DailyRate": 200,
            "HourlyRate": 20,
            "StockOptionLevel": 1,
            "PerformanceRating": 3,
            "TrainingTimesLastYear": 0,
            "YearsWithCurrManager": 0,
            "OverallSatisfaction": 2,
        },
    )

    assert response.status_code == 200
    assert "predictions" in response.json()

    predictions = response.json()["predictions"]
    assert len(predictions) == 4
    for prediction in predictions:
        assert "model_name" in prediction
        assert "prediction" in prediction
        assert "attrition_risk" in prediction
        assert prediction["attrition_risk"] in [
            "Faible risque de départ",
            "Risque moyen de départ",
            "Risque élevé de départ",
        ]


def test_get_user_info():
    """Test to retrieve user info with valid token."""
    token_response = client.post(
        "/token", data={"username": "ADMIN", "password": "admin"}
    )
    token = token_response.json()["access_token"]

    response = client.get("/user_info", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    assert "username" in response.json()
    assert response.json()["username"] == "ADMIN"


def test_get_user_info_unauthorized():
    """Test to access user info without authentication."""
    response = client.get("/user_info")

    assert response.status_code == 401


if __name__ == "__main__":
    pytest.main([__file__])
