import os
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import psycopg2
from psycopg2 import pool
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mlops-api")

app = FastAPI(
    title="MLOps Pipeline API",
    description="API de prédiction du diabète",
    version="2.2.0"
)

MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"
METRICS_PATH = "metrics.json"

DEMO_HTML_PATH = Path(__file__).resolve().parent.parent / "templates" / "predict.html"

# POSTGRES

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/mlops_predictions"
)

db_pool = None
try:
    db_pool = psycopg2.pool.SimpleConnectionPool(1, 10, DATABASE_URL)
    logger.info("Pool de connexions Postgres initialisé")
except Exception as e:
    # On NE PLANTE PAS l'API si Postgres est down : les prédictions doivent
    # continuer à fonctionner, on perd juste le logging.
    logger.warning(f"Postgres indisponible au démarrage ({e}) — logging désactivé")


def init_db():
    if db_pool is None:
        return
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    pregnancies INT,
                    glucose FLOAT,
                    blood_pressure FLOAT,
                    skin_thickness FLOAT,
                    insulin FLOAT,
                    bmi FLOAT,
                    diabetes_pedigree FLOAT,
                    age INT,
                    prediction INT,
                    confidence FLOAT,
                    algorithm TEXT
                );
            """)
        conn.commit()
        logger.info("Table 'predictions' prête")
    except Exception as e:
        logger.warning(f"Impossible d'initialiser la table predictions : {e}")
    finally:
        db_pool.putconn(conn)


def log_prediction(data: "DiabetesInput", prediction: int, confidence: float, algorithm: str):
    if db_pool is None:
        return
    conn = None
    try:
        conn = db_pool.getconn()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO predictions
                (pregnancies, glucose, blood_pressure, skin_thickness, insulin,
                 bmi, diabetes_pedigree, age, prediction, confidence, algorithm)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                data.pregnancies, data.glucose, data.blood_pressure, data.skin_thickness,
                data.insulin, data.bmi, data.diabetes_pedigree, data.age,
                prediction, confidence, algorithm
            ))
        conn.commit()
    except Exception as e:
        logger.warning(f"Échec du log de prédiction dans Postgres : {e}")
    finally:
        if conn:
            db_pool.putconn(conn)


# MODÈLE
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


def load_demo_html():
    if DEMO_HTML_PATH.exists():
        return DEMO_HTML_PATH.read_text(encoding="utf-8")
    return "<h1>Page de démo introuvable</h1>"


model = load_model()
scaler = load_scaler()
metrics_info = load_metrics()
demo_html = load_demo_html()
init_db()

# Prometheus METRICS
Instrumentator().instrument(app).expose(app)


# SCHEMAS
class DiabetesInput(BaseModel):
    pregnancies: int = Field(ge=0, le=20)
    glucose: float = Field(gt=0, le=300)
    blood_pressure: float = Field(gt=0, le=200)
    skin_thickness: float = Field(ge=0, le=100)
    insulin: float = Field(ge=0, le=1000)
    bmi: float = Field(gt=0, le=80)
    diabetes_pedigree: float = Field(gt=0, le=3)
    age: int = Field(ge=1, le=120)


class PredictionOutput(BaseModel):
    prediction: int
    result: str
    confidence: float
    algorithm: str = "unknown"


# ROUTES
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
        "db_connected": db_pool is not None,
        "algorithm": metrics_info.get("best_algorithm", "unknown")
    }


@app.get("/stats")
def stats():
    if db_pool is None:
        raise HTTPException(status_code=503, detail="Postgres indisponible")
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*), AVG(confidence), SUM(prediction) FROM predictions;")
            total, avg_confidence, total_positive = cur.fetchone()
        return {
            "total_predictions": total or 0,
            "avg_confidence": round(float(avg_confidence), 4) if avg_confidence else None,
            "total_predicted_diabetic": total_positive or 0
        }
    finally:
        db_pool.putconn(conn)


@app.get("/predict", response_class=HTMLResponse)
def demo():
    return demo_html


@app.post("/predire", response_model=PredictionOutput)
def predict(data: DiabetesInput):
    try:
        features = np.array([[
            data.pregnancies, data.glucose, data.blood_pressure, data.skin_thickness,
            data.insulin, data.bmi, data.diabetes_pedigree, data.age
        ]])

        if scaler is not None and metrics_info.get("needs_scaling", False):
            features = scaler.transform(features)

        prediction = int(model.predict(features)[0])
        confidence = float(model.predict_proba(features)[0][prediction])
        algorithm = metrics_info.get("best_algorithm", "unknown")

        # Le logging ne doit jamais faire échouer la réponse à l'utilisateur
        log_prediction(data, prediction, confidence, algorithm)

        return PredictionOutput(
            prediction=prediction,
            result="Diabétique" if prediction == 1 else "Non diabétique",
            confidence=round(confidence, 4),
            algorithm=algorithm
        )
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Entrée invalide : {e}")
    except Exception as e:
        logger.error(f"Erreur inattendue lors de la prédiction : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")