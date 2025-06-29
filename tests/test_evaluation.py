import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, os.pardir)
sys.path.insert(0, parent_dir)

from pipeline_script import train_model, evaluate_model, load_data, preprocess_data
import joblib

@pytest.fixture(scope="module")
def trained_model_and_data():
    df, _ = load_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and save model to a temporary file
    model_path = "temp_test_model.joblib"
    model = train_model(X_train, y_train, model_path)
    yield model, X_test, y_test

    # Clean up the temporary model file
    if os.path.exists(model_path):
        os.remove(model_path)

def test_model_training_saves_model(trained_model_and_data):
    # The fixture already handles training and saving, just check if the file exists
    assert os.path.exists("temp_test_model.joblib")

    # Also check if it's a valid model file
    loaded_model = joblib.load("temp_test_model.joblib")
    assert isinstance(loaded_model, RandomForestClassifier)

def test_evaluation_metrics_are_valid(trained_model_and_data):
    model, X_test, y_test = trained_model_and_data
    metrics = evaluate_model(model, X_test, y_test)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics

    # Check if metrics are within a reasonable range (0 to 1)
    for metric in metrics.values():
        assert 0 <= metric <= 1

    # For Iris, accuracy should be relatively high
    assert metrics["accuracy"] > 0.8