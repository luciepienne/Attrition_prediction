from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from typing import List, Optional
import json
import pickle
import numpy as np

app = FastAPI()

# Charger les informations sur les features
with open('models/feature_info_knn.json', 'r') as f:
    feature_info = json.load(f)

# Charger le modèle
with open('models/knn_model.pkl', 'rb') as f:
    model = pickle.load(f)

class PredictionInput(BaseModel):
    Age: int = Field(..., ge=18, le=65, description="Age of the employee (18-65)")
    BusinessTravel: str = Field(..., description="Frequency of business travel")
    DailyRate: int = Field(..., ge=0, description="Daily rate of pay")
    Department: str = Field(..., description="Department of the employee")
    DistanceFromHome: float = Field(..., ge=0, description="Distance from home to work (in miles)")
    Education: int = Field(..., ge=1, le=5, description="1 'Below College' 2 'College' 3 'Bachelor' 4 'Master' 5 'Doctor'")
    EducationField: str = Field(..., description="Field of education")
    Gender: str = Field(..., description="Gender of the employee")
    HourlyRate: int = Field(..., ge=0, description="Hourly rate of pay")
    JobInvolvement: int = Field(..., ge=1, le=4, description="Job involvement level (1-4)")
    JobLevel: int = Field(..., ge=1, le=5, description="Job level in the company (1-5)")
    JobRole: str = Field(..., description="Current role in the company")
    MaritalStatus: str = Field(..., description="Marital status of the employee")
    MonthlyIncome: float = Field(..., ge=0, description="Monthly income")
    MonthlyRate: int = Field(..., ge=0, description="Monthly rate")
    NumCompaniesWorked: int = Field(..., ge=0, description="Number of companies worked at")
    OverTime: str = Field(..., description="Whether the employee works overtime")
    PercentSalaryHike: float = Field(..., ge=0, le=100, description="Percentage of salary hike")
    PerformanceRating: int = Field(..., ge=1, le=4, description="Performance rating (1-4)")
    StockOptionLevel: int = Field(..., ge=0, le=3, description="Stock option level (0-3)")
    TrainingTimesLastYear: int = Field(..., ge=0, description="Number of training sessions attended last year")
    YearsWithCurrManager: float = Field(..., ge=0, description="Years with current manager")
    WorkExperience: float = Field(..., ge=0, description="Total years of work experience")
    OverallSatisfaction: float = Field(..., ge=1, le=5, description="Overall job satisfaction (1-5)")

class PredictionOutput(BaseModel):
    prediction: float
    attrition_risk: str

@app.post("/predict", response_model=PredictionOutput)
async def predict(
    Gender: str = Query(..., description="Gender of the employee", enum=list(feature_info['encoding_dict']['Gender'].keys())),
    Age: int = Query(..., ge=18, le=65, description="Age of the employee (18-65)"),
    MaritalStatus: str = Query(..., description="Marital status of the employee", enum=list(feature_info['encoding_dict']['MaritalStatus'].keys())),
    Education: int = Query(..., ge=1, le=5, description="1 'Below College' 2 'College' 3 'Bachelor' 4 'Master' 5 'Doctor'"),
    EducationField: str = Query(..., description="Field of education", enum=list(feature_info['encoding_dict']['EducationField'].keys())),
    NumCompaniesWorked: int = Query(..., ge=0, description="Number of companies worked at"),
    WorkExperience: float = Query(..., ge=0, description="Total years of work experience"),
    Department: str = Query(..., description="Department of the employee", enum=list(feature_info['encoding_dict']['Department'].keys())),
    JobRole: str = Query(..., description="Current role in the company", enum=list(feature_info['encoding_dict']['JobRole'].keys())),
    JobLevel: int = Query(..., ge=1, le=5, description="Job level in the company (1-5)"),
    DistanceFromHome: float = Query(..., ge=0, description="Distance from home to work (in miles)"),
    BusinessTravel: str = Query(..., description="Frequency of business travel", enum=list(feature_info['encoding_dict']['BusinessTravel'].keys())),
    OverTime: str = Query(..., description="Whether the employee works overtime", enum=list(feature_info['encoding_dict']['OverTime'].keys())),
    JobInvolvement: int = Query(..., ge=1, le=4, description="Job involvement level (1-4)"),
    PercentSalaryHike: float = Query(..., ge=0, le=100, description="Percentage of salary hike"),
    MonthlyIncome: float = Query(..., ge=0, description="Monthly income"),
    MonthlyRate: int = Query(..., ge=0, description="Monthly rate"),
    DailyRate: int = Query(..., ge=0, description="Daily rate of pay"),
    HourlyRate: int = Query(..., ge=0, description="Hourly rate of pay"),
    StockOptionLevel: int = Query(..., ge=0, le=3, description="Stock option level (0-3)"),
    PerformanceRating: int = Query(..., ge=1, le=4, description="Performance rating (1-4)"),
    TrainingTimesLastYear: int = Query(..., ge=0, description="Number of training sessions attended last year"),
    YearsWithCurrManager: float = Query(..., ge=0, description="Years with current manager"),
    OverallSatisfaction: float = Query(..., ge=1, le=5, description="Overall job satisfaction (1-5)")
):
    # Créer un dictionnaire avec les entrées
    input_data = locals()
    
    # Convertir les entrées en valeurs numériques selon l'encodage
    features = []
    for feature_name in feature_info['feature_names']:
        value = input_data[feature_name]
        if feature_name in feature_info['encoding_dict']:
            value = feature_info['encoding_dict'][feature_name].get(value, value)
        features.append(float(value))  # Convertir en float pour s'assurer que c'est numérique

    # Faire la prédiction
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict_proba(features_array)[0][1]  # Probabilité de départ

    # Interpréter la prédiction
    if prediction < 0.3:
        risk = "Faible risque de départ"
    elif prediction < 0.7:
        risk = "Risque moyen de départ"
    else:
        risk = "Risque élevé de départ"

    return PredictionOutput(prediction=float(prediction), attrition_risk=risk)


@app.get("/feature_options")
async def get_feature_options():
    return feature_info['encoding_dict']

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)