from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from enum import Enum
import pickle
import numpy as np

app = FastAPI()

# Charger le modèle
with open('models/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

class Gender(str, Enum):
    male = "Male"
    female = "Female"

class PredictionInput(BaseModel):
    age: int = Field(..., ge=18, le=65, description="Age of the employee")
    gender: Gender
    # Ajoutez d'autres champs ici si nécessaire

class PredictionOutput(BaseModel):
    attrition_probability: float
    risk_category: str
    prediction: str

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        # Préparer les données pour la prédiction
        gender_encoded = 1 if input_data.gender == Gender.male else 0
        
        features = np.array([[
            input_data.age,
            gender_encoded,
            # Ajoutez d'autres caractéristiques ici avec des valeurs par défaut
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]])
        
        # Faire la prédiction
        prediction = model.predict_proba(features)[0]
        
        # Interpréter la prédiction
        attrition_probability = prediction[1]  # Probabilité de départ
        
        if attrition_probability < 0.3:
            risk_category = "Faible risque de départ"
        elif attrition_probability < 0.7:
            risk_category = "Risque moyen de départ"
        else:
            risk_category = "Risque élevé de départ"

        return PredictionOutput(
            attrition_probability=float(attrition_probability),
            risk_category=risk_category,
            prediction="Susceptible de partir" if attrition_probability > 0.5 else "Susceptible de rester"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)