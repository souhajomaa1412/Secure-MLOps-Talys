import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import joblib
import os
import json

# ========== CONFIGURATION ==========
DATASET_PATH = 'data/diabetes.csv'
MODEL_PATH = 'models/model.pkl'
METRICS_PATH = 'metrics.json'
SEUIL_ACCEPTATION = 0.75

# Charger le dataset
df = pd.read_csv(DATASET_PATH)
print(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# MLflow tracking
mlflow.set_experiment("mlops-pipeline-diabetes")

with mlflow.start_run():

    # GridSearch
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    grid = GridSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=42),
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    print(f"Meilleurs paramètres : {grid.best_params_}")

    # Évaluation sur test set
    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    # Logger dans MLflow
    mlflow.log_param("best_n_estimators", grid.best_params_['n_estimators'])
    mlflow.log_param("best_max_depth", str(grid.best_params_['max_depth']))
    mlflow.log_param("best_min_samples_split", grid.best_params_['min_samples_split'])
    mlflow.log_metric("best_cv_f1", grid.best_score_)
    mlflow.log_metric("accuracy", float(accuracy))
    mlflow.log_metric("precision", float(precision))
    mlflow.log_metric("recall", float(recall))
    mlflow.log_metric("f1_score", float(f1))
    mlflow.sklearn.log_model(best_model, "model")

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"Run ID    : {mlflow.active_run().info.run_id}")

    # ✅ Sauvegarder le modèle
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"Modèle sauvegardé dans {MODEL_PATH}")

    # ✅ Sauvegarder les métriques pour DVC + validate.py
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "best_n_estimators": grid.best_params_['n_estimators'],
        "best_max_depth": str(grid.best_params_['max_depth']),
        "best_min_samples_split": grid.best_params_['min_samples_split'],
        "dataset": "diabetes",
        "threshold": SEUIL_ACCEPTATION
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Métriques sauvegardées dans {METRICS_PATH}")