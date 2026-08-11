import pytest
import pandas as pd
import numpy as np
import joblib
import os
import json
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

# TESTS API
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_page_returns_html():
    response = client.get("/predict")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_predire_valid():
    response = client.post("/predire", json={
        "pregnancies": 2,
        "glucose": 120.0,
        "blood_pressure": 70.0,
        "skin_thickness": 25.0,
        "insulin": 80.0,
        "bmi": 28.5,
        "diabetes_pedigree": 0.5,
        "age": 35
    })
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in [0, 1]
    assert "confidence" in data
    assert 0 <= data["confidence"] <= 1
    assert "result" in data


def test_predire_missing_field():
    response = client.post("/predire", json={
        "glucose": 120.0
    })
    assert response.status_code == 422


def test_predire_result_values():
    response = client.post("/predire", json={
        "pregnancies": 2,
        "glucose": 120.0,
        "blood_pressure": 70.0,
        "skin_thickness": 25.0,
        "insulin": 80.0,
        "bmi": 28.5,
        "diabetes_pedigree": 0.5,
        "age": 35
    })
    data = response.json()
    assert data["result"] in ["Diabétique", "Non diabétique"]


def test_predire_out_of_range_rejected():
    response = client.post("/predire", json={
        "pregnancies": 999,   # hors limite (le=20)
        "glucose": 120.0,
        "blood_pressure": 70.0,
        "skin_thickness": 25.0,
        "insulin": 80.0,
        "bmi": 28.5,
        "diabetes_pedigree": 0.5,
        "age": 35
    })
    assert response.status_code == 422


# TESTS DONNÉES
def test_dataset_exists():
    assert os.path.exists("data/diabetes.csv"), "Dataset introuvable"


def test_dataset_no_nulls():
    df = pd.read_csv("data/diabetes.csv")
    assert df.isnull().sum().sum() == 0


def test_target_column():
    df = pd.read_csv("data/diabetes.csv")
    assert "Outcome" in df.columns
    assert set(df["Outcome"].unique()) == {0, 1}


# TESTS MODÈLE
def test_model_exists():
    assert os.path.exists("models/model.pkl")


def test_model_predict():
    model = joblib.load("models/model.pkl")
    df = pd.read_csv("data/diabetes_processed.csv")
    sample = df.drop("Outcome", axis=1).iloc[[0]]
    pred = model.predict(sample)
    assert pred[0] in [0, 1]


def test_model_accuracy_on_holdout():

    model = joblib.load("models/model.pkl")
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv").iloc[:, 0]
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, model.predict(X_test))
    assert acc >= 0.60, f"Accuracy trop faible sur le holdout : {acc:.2f}"


# TESTS MÉTRIQUES
def test_metrics_file_exists():
    assert os.path.exists("metrics.json")


def test_metrics_content():
    with open("metrics.json") as f:
        metrics = json.load(f)
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert metrics["f1_score"] >= metrics.get("threshold", 0.65)