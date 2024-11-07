"""
This module contains the main FastAPI application for employee attrition prediction.
It includes endpoints for authentication and prediction using various machine learning models.
"""

from typing import List
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import JWTError, jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext
import pickle

# Constants
SECRET_KEY = "your-secret-key"  # Change this!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize FastAPI app
app = FastAPI()

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Load models
with open('models/knn_model.pkl', 'rb') as f:
    KNN_MODEL = pickle.load(f)

with open('models/random_forest_model.pkl', 'rb') as f:
    RF_MODEL = pickle.load(f)

with open('models/xgboost_model.pkl', 'rb') as f:
    XGB_MODEL = pickle.load(f)

with open('models/linear_regression_model.pkl', 'rb') as f:
    LR_MODEL = pickle.load(f)

class Token(BaseModel):
    """
    Pydantic model for token response.
    """
    access_token: str
    token_type: str

class User(BaseModel):
    """
    Pydantic model for user.
    """
    username: str

class PredictionInput(BaseModel):
    """
    Pydantic model for prediction input.
    """
    model: str
    features: List[float]

def create_access_token(data: dict) -> str:
    """
    Create a new access token.

    Args:
        data (dict): Data to encode in the token.

    Returns:
        str: Encoded JWT token.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    Get the current user from the token.

    Args:
        token (str): JWT token.

    Returns:
        User: User object.

    Raises:
        HTTPException: If credentials are invalid.
    """
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        return User(username=username)
    except JWTError:
        raise credentials_exception

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
    # This is a simple check. In a real app, you'd check against a database.
    if (form_data.username != "admin" or
            not pwd_context.verify(form_data.password, pwd_context.hash("password"))):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/predict")
async def predict(input_data: PredictionInput, current_user: User = Depends(get_current_user)):
    """
    Make a prediction using the specified model.

    Args:
        input_data (PredictionInput): Input data for prediction.
        current_user (User): Current authenticated user.

    Returns:
        dict: Prediction result.

    Raises:
        HTTPException: If an invalid model is specified.
    """
    if input_data.model == 'knn':
        prediction = KNN_MODEL.predict([input_data.features])
    elif input_data.model == 'rf':
        prediction = RF_MODEL.predict([input_data.features])
    elif input_data.model == 'xgb':
        prediction = XGB_MODEL.predict([input_data.features])
    elif input_data.model == 'lr':
        prediction = LR_MODEL.predict([input_data.features])
    else:
        raise HTTPException(status_code=400, detail="Invalid model name")

    return {"prediction": prediction.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)