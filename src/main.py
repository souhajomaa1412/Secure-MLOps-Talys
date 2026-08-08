from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
import json

app = FastAPI(
    title="MLOps Pipeline API",
    description="API de prédiction du diabète",
    version="2.1.0"
)

MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"
METRICS_PATH = "metrics.json"


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def load_scaler():
    if os.path.exists(SCALER_PATH):
        return joblib.load(SCALER_PATH)
    return None


def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}


model = load_model()
scaler = load_scaler()
metrics_info = load_metrics()


class DiabetesInput(BaseModel):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree: float
    age: int


class PredictionOutput(BaseModel):
    prediction: int
    result: str
    confidence: float
    algorithm: str = "unknown"


@app.get("/")
def root():
    return {
        "message": "MLOps Pipeline API - Diabète",
        "status": "running",
        "model_algorithm": metrics_info.get("best_algorithm", "unknown")
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "algorithm": metrics_info.get("best_algorithm", "unknown")
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(data: DiabetesInput):
    try:
        features = np.array([[
            data.pregnancies,
            data.glucose,
            data.blood_pressure,
            data.skin_thickness,
            data.insulin,
            data.bmi,
            data.diabetes_pedigree,
            data.age
        ]])

        if scaler is not None and metrics_info.get("needs_scaling", False):
            features = scaler.transform(features)

        prediction = int(model.predict(features)[0])
        confidence = float(model.predict_proba(features)[0][prediction])

        return PredictionOutput(
            prediction=prediction,
            result="Diabétique" if prediction == 1 else "Non diabétique",
            confidence=round(confidence, 4),
            algorithm=metrics_info.get("best_algorithm", "unknown")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))