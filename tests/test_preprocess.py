""" Unit tests for key functions to prepare data for training"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing.preprocess_and_split import (
    apply_smote,
    encode_categorical,
    scale_features,
    split_data,
)


def test_encode_categorical():
    df = pd.DataFrame(
        {"Color": ["Red", "Green", "Blue", "Green"], "Value": [1, 2, 3, 4]}
    )

    encoded_df, encoding_dict = encode_categorical(df)

    assert "Color" in encoded_df.columns
    assert all(encoded_df["Color"].isin([0, 1, 2]))
    assert encoding_dict["Color"] == {"Blue": 0, "Green": 1, "Red": 2}


def test_split_data():
    df = pd.DataFrame({"Attrition": [0, 1, 0, 1], "Feature1": [10, 20, 30, 40]})

    X_train, X_test, y_train, y_test = split_data(df)

    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)


def test_scale_features():
    X_train = np.array([[1.0], [2.0], [3.0]])
    X_test = np.array([[4.0], [5.0]])

    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    assert np.allclose(X_train_scaled.mean(), 0)
    assert np.allclose(X_train_scaled.std(), 1)


def test_apply_smote():
    X_train = np.array([[1], [2], [3], [4]])
    y_train = np.array([0, 0, 1, 1])  # Imbalanced classes

    X_resampled, y_resampled = apply_smote(X_train, y_train)

    assert len(y_resampled) >= len(y_train)
    assert np.unique(y_resampled).size == 2  # Should have both classes


if __name__ == "__main__":
    pytest.main([__file__])
