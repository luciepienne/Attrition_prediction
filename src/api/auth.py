"""Module defining rules for authentification in API"""

from datetime import datetime, timedelta

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

SECRET_KEY = "password"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# base de données d'utilisateurs, simulée
fake_users_db = {
    "ADMIN": {
        "username": "admin",
        "full_name": "Admin Master",
        "email": "master@example.com",
        "hashed_password": pwd_context.hash("admin"),
    }
}


class Token(BaseModel):
    """Class to define token"""

    access_token: str
    token_type: str


class User(BaseModel):
    """Class to define users"""

    username: str


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


def authenticate_user(fake_db, username: str, password: str):
    """
    Authenticate user against fake database.

    Args:
        fake_db (dict): Fake user database.
        username (str): Username to authenticate.
        password (str): Password to authenticate.

    Returns:
        User or False: User object if authentication is successful; False otherwise.
    """

    user = fake_db.get(username)

    if not user or not pwd_context.verify(password, user["hashed_password"]):
        return False

    return User(username=username)


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
