""" Unit tests for key functions to train and chose best model"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.model_training.train_and_best import train_and_log_model


@pytest.fixture(autouse=True)
def mock_mlflow():
    with patch("mlflow.set_experiment"), patch("mlflow.start_run"), patch(
        "mlflow.log_param"
    ), patch("mlflow.log_metric"), patch("mlflow.sklearn.log_model"), patch(
        "mlflow.end_run"
    ):
        yield


@patch("mlflow.log_metric")
@patch("mlflow.log_param")
@patch("mlflow.sklearn.log_model")
def test_train_and_log_model(mock_log_model, mock_log_param, mock_log_metric):
    # Créer des données synthétiques pour le test
    X = np.random.rand(100, 20)  # 100 samples, 20 features
    y = np.random.randint(0, 2, size=100)  # Binary target variable (0 or 1)

    # Créer un modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Appeler la fonction train_and_log_model
    accuracy = train_and_log_model(model, "Random Forest", X, y, X, y)

    # Vérifier que l'accuracy est dans une plage raisonnable
    assert 0 <= accuracy <= 1.0

    # Vérifier que les méthodes MLflow ont été appelées
    mock_log_param.assert_called()
    mock_log_metric.assert_called()
    mock_log_model.assert_called()

    # Vérifier que certaines métriques spécifiques ont été loggées
    mock_log_metric.assert_any_call("test_accuracy", pytest.approx(accuracy, abs=1e-6))
