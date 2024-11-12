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
from pydantic import BaseModel, Field, ConfigDict
from prometheus_client import start_http_server, Summary, Counter
import mlflow
from contextlib import asynccontextmanager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Métriques Prometheus
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
PREDICTION_COUNTER = Counter('prediction_count', 'Number of predictions made', ['model', 'risk_level'])

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
    class Config:
        protected_namespaces = ()
    def dict(self, *args, **kwargs):
        return {
            "model_name": self.model_name,
            "prediction": self.prediction,
            "attrition_risk": self.attrition_risk
        }

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
    else:
        raise ValueError(f"Model {model_name} not supported")

    # Interpréter la prédiction
    if prediction_proba < 0.3:
        risk = "Faible risque de départ"
    elif prediction_proba < 0.7:
        risk = "Risque moyen de départ"
    else:
        risk = "Risque élevé de départ"

    return {
        "model_name": model_used,
        "prediction": float(prediction_proba),
        "attrition_risk": risk
    }


@app.post("/predict", response_model=dict)
@REQUEST_TIME.time()
def predict(input_data: PredictionInput, current_user: User = Depends(get_current_user)):
    # Convertir les entrées en valeurs numériques selon l'encodage
    features = []
    for feature_name in feature_info["feature_names"]:
        value = getattr(input_data, feature_name)
        if feature_name in feature_info["encoding_dict"]:
            value = feature_info["encoding_dict"][feature_name].get(value, value)
        features.append(float(value))

    # Faire la prédiction pour chaque modèle
    features_array = np.array(features).reshape(1, -1)
    predictions_output = {"predictions": []}

    # Liste des modèles à utiliser
    models = ["knn", "random_forest", "linear_regression", "xg_boost"]

    for model_name in models:
        try:
            prediction = predict_model(model_name, features_array)
            predictions_output["predictions"].append(prediction)
               
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction avec le modèle {model_name}: {str(e)}")
    
    PREDICTION_COUNTER.labels(model=prediction["model_name"], risk_level=prediction["attrition_risk"]).inc()
    mlflow.log_metric(f"{prediction['model_name']}_prediction", prediction["prediction"])

    logger.info(f"Prediction output: {predictions_output}")

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

    uvicorn.run(app, host="0.0.0.0", port=8001)
