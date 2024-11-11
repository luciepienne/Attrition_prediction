"""
This module contains the main FastAPI application for employee attrition prediction.
It includes endpoints for authentication and prediction using various machine learning models, 
features necessary for prediction are integrated.
"""

import json
import os
import pickle
import sys


import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth import (
    Token,
    User,
    authenticate_user,
    create_access_token,
    fake_users_db,
    get_current_user,
)

app = FastAPI()

# Charger les informations sur les features
with open("models/feature_info.json", "r") as f:
    feature_info = json.load(f)

# Charger tous les modèles
with open("models/knn_model.pkl", "rb") as f:
    knn_model = pickle.load(f)

with open("models/random_forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("models/linear_regression_model.pkl", "rb") as f:
    lr_model = pickle.load(f)

with open("models/xgboost_model.pkl", "rb") as f:
    xgboost_model = pickle.load(f)


class PredictionInput(BaseModel):
    """
    Pydantic model for prediction input.
    """

    Age: int = Field(..., ge=18, le=65, description="Age of the employee (18-65)")
    BusinessTravel: str = Field(..., description="Frequency of business travel")
    DailyRate: int = Field(..., ge=0, description="Daily rate of pay")
    Department: str = Field(..., description="Department of the employee")
    DistanceFromHome: float = Field(
        ..., ge=0, description="Distance from home to work (in miles)"
    )
    Education: int = Field(
        ...,
        ge=1,
        le=5,
        description="1 'Below College' 2 'College' 3 'Bachelor' 4 'Master' 5 'Doctor'",
    )
    EducationField: str = Field(..., description="Field of education")
    Gender: str = Field(..., description="Gender of the employee")
    HourlyRate: int = Field(..., ge=0, description="Hourly rate of pay")
    JobInvolvement: int = Field(
        ..., ge=1, le=4, description="Job involvement level (1-4)"
    )
    JobLevel: int = Field(..., ge=1, le=5, description="Job level in the company (1-5)")
    JobRole: str = Field(..., description="Current role in the company")
    MaritalStatus: str = Field(..., description="Marital status of the employee")
    MonthlyIncome: float = Field(..., ge=0, description="Monthly income")
    MonthlyRate: int = Field(..., ge=0, description="Monthly rate")
    NumCompaniesWorked: int = Field(
        ..., ge=0, description="Number of companies worked at"
    )
    OverTime: str = Field(..., description="Whether the employee works overtime")
    PercentSalaryHike: float = Field(
        ..., ge=0, le=100, description="Percentage of salary hike"
    )
    PerformanceRating: int = Field(
        ..., ge=1, le=4, description="Performance rating (1-4)"
    )
    StockOptionLevel: int = Field(
        ..., ge=0, le=3, description="Stock option level (0-3)"
    )
    TrainingTimesLastYear: int = Field(
        ..., ge=0, description="Number of training sessions attended last year"
    )
    YearsWithCurrManager: float = Field(
        ..., ge=0, description="Years with current manager"
    )
    WorkExperience: float = Field(
        ..., ge=0, description="Total years of work experience"
    )
    OverallSatisfaction: float = Field(
        ..., ge=1, le=5, description="Overall job satisfaction (1-5)"
    )


class PredictionOutput(BaseModel):
    model_name: str
    prediction: float
    attrition_risk: str


@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate user and return a token.

    Args:
        form_data (OAuth2PasswordRequestForm): Form data containing username and password.

    Returns:
        dict: Token information.

    Raises:
        HTTPException: If authentication fails.
    """

    user = authenticate_user(fake_users_db, form_data.username, form_data.password)

    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token = create_access_token(data={"sub": user.username})

    return {"access_token": access_token, "token_type": "bearer"}


def predict_model(model_name: str, features_array):
    """
    Fonction pour faire la prédiction avec un modèle donné.

    Args:
        model_name (str): Le nom du modèle à utiliser.
        features_array (np.array): Les features à utiliser pour la prédiction.

    Returns:
        PredictionOutput: Un objet contenant le nom du modèle et la prédiction.
    """

    if model_name == "knn":
        prediction_proba = knn_model.predict_proba(features_array)[0][1]
        model_used = "KNN"

    elif model_name == "random_forest":
        prediction_proba = rf_model.predict_proba(features_array)[0][1]
        model_used = "Random Forest"

    elif model_name == "linear_regression":
        prediction_proba = lr_model.predict_proba(features_array)[0][1]
        model_used = "Linear Regression"

    elif model_name == "xg_boost":
        prediction_proba = xgboost_model.predict_proba(features_array)[0][1]
        model_used = "XG BOOST"

    # Interpréter la prédiction
    if prediction_proba < 0.3:
        risk = "Faible risque de départ"
    elif prediction_proba < 0.7:
        risk = "Risque moyen de départ"
    else:
        risk = "Risque élevé de départ"

    return PredictionOutput(
        model_name=model_used, prediction=float(prediction_proba), attrition_risk=risk
    )


@app.post("/predict", response_model=dict)
async def predict(
    Gender: str = Query(
        ...,
        description="Gender of the employee",
        enum=list(feature_info["encoding_dict"]["Gender"].keys()),
    ),
    Age: int = Query(..., ge=18, le=65, description="Age of the employee (18-65)"),
    MaritalStatus: str = Query(
        ...,
        description="Marital status of the employee",
        enum=list(feature_info["encoding_dict"]["MaritalStatus"].keys()),
    ),
    Education: int = Query(
        ...,
        ge=1,
        le=5,
        description="1 'Below College' 2 'College' 3 'Bachelor' 4 'Master' 5 'Doctor'",
    ),
    EducationField: str = Query(
        ...,
        description="Field of education",
        enum=list(feature_info["encoding_dict"]["EducationField"].keys()),
    ),
    NumCompaniesWorked: int = Query(
        ..., ge=0, description="Number of companies worked at"
    ),
    WorkExperience: float = Query(
        ..., ge=0, description="Total years of work experience"
    ),
    Department: str = Query(
        ...,
        description="Department of the employee",
        enum=list(feature_info["encoding_dict"]["Department"].keys()),
    ),
    JobRole: str = Query(
        ...,
        description="Current role in the company",
        enum=list(feature_info["encoding_dict"]["JobRole"].keys()),
    ),
    JobLevel: int = Query(
        ..., ge=1, le=5, description="Job level in the company (1-5)"
    ),
    DistanceFromHome: float = Query(
        ..., ge=0, description="Distance from home to work (in miles)"
    ),
    BusinessTravel: str = Query(
        ...,
        description="Frequency of business travel",
        enum=list(feature_info["encoding_dict"]["BusinessTravel"].keys()),
    ),
    OverTime: str = Query(
        ...,
        description="Whether the employee works overtime",
        enum=list(feature_info["encoding_dict"]["OverTime"].keys()),
    ),
    JobInvolvement: int = Query(
        ..., ge=1, le=4, description="Job involvement level (1-4)"
    ),
    PercentSalaryHike: float = Query(
        ..., ge=0, le=100, description="Percentage of salary hike"
    ),
    MonthlyIncome: float = Query(..., ge=0, description="Monthly income"),
    MonthlyRate: int = Query(..., ge=0, description="Monthly rate"),
    DailyRate: int = Query(..., ge=0, description="Daily rate of pay"),
    HourlyRate: int = Query(..., ge=0, description="Hourly rate of pay"),
    StockOptionLevel: int = Query(
        ..., ge=0, le=3, description="Stock option level (0-3)"
    ),
    PerformanceRating: int = Query(
        ..., ge=1, le=4, description="Performance rating (1-4)"
    ),
    TrainingTimesLastYear: int = Query(
        ..., ge=0, description="Number of training sessions attended last year"
    ),
    YearsWithCurrManager: float = Query(
        ..., ge=0, description="Years with current manager"
    ),
    OverallSatisfaction: float = Query(
        ..., ge=1, le=5, description="Overall job satisfaction (1-5)"
    ),
    current_user: User = Depends(get_current_user),
):

    # Créer un dictionnaire avec les entrées
    input_dict = locals()

    # Convertir les entrées en valeurs numériques selon l'encodage
    features = []

    for feature_name in feature_info["feature_names"]:
        value = input_dict[feature_name]
        if feature_name in feature_info["encoding_dict"]:
            value = feature_info["encoding_dict"][feature_name].get(value)
        features.append(float(value))

    # Faire la prédiction pour chaque modèle
    features_array = np.array(features).reshape(1, -1)

    predictions_output = {"predictions": []}

    # Prédictions pour chaque modèle
    predictions_output["predictions"].append(predict_model("knn", features_array))
    predictions_output["predictions"].append(
        predict_model("random_forest", features_array)
    )
    predictions_output["predictions"].append(
        predict_model("linear_regression", features_array)
    )
    predictions_output["predictions"].append(predict_model("xg_boost", features_array))

    return predictions_output


@app.get("/user_info")
async def get_user_info(current_user: User = Depends(get_current_user)):
    """
    Récupère les informations de l'utilisateur actuellement authentifié.

    Cette fonction nécessite une authentification et renvoie simplement le nom d'utilisateur
    de l'utilisateur actuellement connecté.
    """
    return {"username": current_user.username}


@app.get("/docs", include_in_schema=False)
async def get_docs():
    """Serve the API documentation."""
    return get_swagger_ui_html(openapi_url=app.openapi_url, title="API docs")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
