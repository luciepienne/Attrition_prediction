"""
This module contains the main FastAPI application for employee attrition prediction.
It includes endpoints for authentication and prediction using various machine learning models, 
features necessary for prediction are integrated.
"""

import json
import logging
import os
import pickle
import sys
from contextlib import asynccontextmanager

import mlflow
import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.security import OAuth2PasswordRequestForm
from prometheus_client import Counter, Summary, start_http_server
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.auth import (
    Token,
    User,
    authenticate_user,
    create_access_token,
    fake_users_db,
    get_current_user,
)

app = FastAPI()

# Métriques Prometheus
REQUEST_TIME = Summary("request_processing_seconds", "Time spent processing request")
PREDICTION_COUNTER = Counter(
    "prediction_count", "Number of predictions made", ["model", "risk_level"]
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    start_http_server(8000)  # Démarrer le serveur Prometheus
    mlflow.set_tracking_uri("http://localhost:5000")  # Configurer MLflow
    yield
    # Shutdown


app = FastAPI(lifespan=lifespan)


# Charger les informations sur les features
with open("models/feature_info.json", "r") as f:
    feature_info = json.load(f)

# Charger le nom du meilleur modèle
with open("models/best_model_detail.json", "r") as f:
    best_model_info = json.load(f)

# Charger le meilleur modèle
best_model_name = best_model_info[
    "best_model_name"
]  # Récupérer le nom du meilleur modèle
with open(f"models/best_model_{best_model_name}.pkl", "rb") as f:
    best_model = pickle.load(f)


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
    """Modèle Pydantic pour la sortie de prédiction."""

    best_model_name: str
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


@app.post("/predict", response_model=PredictionOutput)
async def predict(
    input_data: PredictionInput, current_user: User = Depends(get_current_user)
):
    """Faire une prédiction sur l'attrition des employés."""

    # Convertir les entrées en valeurs numériques selon l'encodage
    features = []
    for feature_name in feature_info["feature_names"]:
        value = getattr(input_data, feature_name)
        if feature_name in feature_info["encoding_dict"]:
            value = feature_info["encoding_dict"][feature_name].get(value, value)
        features.append(float(value))

    # Préparer les données pour la prédiction
    features_array = np.array(features).reshape(1, -1)

    # Faire la prédiction avec le meilleur modèle
    prediction_proba = best_model.predict_proba(features_array)[0][
        1
    ]  # Assurez-vous que cela correspond à votre modèle

    # Interpréter la prédiction
    if prediction_proba < 0.3:
        risk = "Faible risque de départ"
    elif prediction_proba < 0.7:
        risk = "Risque moyen de départ"
    else:
        risk = "Risque élevé de départ"

        # Log des métriques avec Prometheus et MLflow
    PREDICTION_COUNTER.labels(model=best_model_name, risk_level=risk).inc()

    # Log de la prédiction dans MLflow (assurez-vous que mlflow est configuré)
    mlflow.log_metric(f"{best_model_name}_prediction", prediction_proba)

    logger.info(
        f"Prediction output: Model={best_model_name}, Probability={prediction_proba}, Risk Level={risk}"
    )

    return PredictionOutput(
        best_model_name=best_model_name,
        prediction=float(prediction_proba),
        attrition_risk=risk,
    )


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

    uvicorn.run(app, host="0.0.0.0", port=8001)
