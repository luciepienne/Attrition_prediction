import os
import sys

import pytest
import requests
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.api.attrition_prediction import get_token


# Simuler un client API
class MockClient:
    def post(self, url, data=None, headers=None):
        if (
            url == "http://localhost:8001/token"
            and data["username"] == "ADMIN"
            and data["password"] == "admin"
        ):
            return MockResponse(
                200, {"access_token": "mock_token", "token_type": "bearer"}
            )
        return MockResponse(400, {"detail": "Invalid credentials"})


class MockResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self.json_data = json_data

    def json(self):
        return self.json_data


def test_login_success(monkeypatch):
    # Remplacer le client par un mock
    monkeypatch.setattr(requests, "post", MockClient().post)

    st.session_state = {}  # Réinitialiser l'état de la session
    username = "ADMIN"
    password = "admin"

    # Simuler le processus de login
    token = get_token(username, password)
    print(token)

    assert token == "mock_token"


def test_login_failure(monkeypatch):
    # Remplacer le client par un mock
    monkeypatch.setattr(requests, "post", MockClient().post)

    st.session_state = {}  # Réinitialiser l'état de la session
    username = "wrong"
    password = "wrong"

    # Simuler le processus de login
    token = get_token(username, password)

    assert token is None  # Vérifiez que le token n'est pas retourné
    assert "token" not in st.session_state  # Vérifiez que rien n'est stocké


if __name__ == "__main__":
    pytest.main([__file__])
