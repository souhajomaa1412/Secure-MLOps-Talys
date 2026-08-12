
import pandas as pd
import numpy as np
import json
import os
import joblib
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow.sklearn

warnings.filterwarnings('ignore')

# CONFIGURATION

DATASET_PATH = os.environ.get('DATASET_PATH', 'data/diabetes_processed.csv')
MODEL_PATH = 'models/model.pkl'
SCALER_PATH = 'models/scaler.pkl'
METRICS_PATH = 'metrics.json'
X_TEST_PATH = 'data/X_test.csv'
Y_TEST_PATH = 'data/y_test.csv'
# Seuil d'acceptation unifié à 0.65 partout (train.py / validate.py /metrics.json)
SEUIL_ACCEPTATION = float(os.environ.get('SEUIL_ACCEPTATION', 0.65))

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"{DATASET_PATH} introuvable. Lance d'abord preprocess_spark.py "
        f"(il génère data/diabetes_processed.csv)."
    )

df = pd.read_csv(DATASET_PATH)
print(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split train/test 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


os.makedirs('data', exist_ok=True)
X_test.to_csv(X_TEST_PATH, index=False)
y_test.to_csv(Y_TEST_PATH, index=False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs('models', exist_ok=True)

# MLFLOW TRACKING
mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("mlops-pipeline-diabetes")
print(f"MLflow tracking URI : {mlflow_uri}")

# DÉFINITION DES 4 MODÈLES
experiments = [
    {
        'name': 'LogisticRegression',
        'model': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        'params': {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']},
        'needs_scaling': True
    },
    {
        'name': 'RandomForest',
        'model': RandomForestClassifier(class_weight='balanced', random_state=42),
        'params': {'n_estimators': [100, 200], 'max_depth': [5, 10, None], 'min_samples_split': [2, 5]},
        'needs_scaling': False
    },
    {
        'name': 'SVM_RBF',
        'model': SVC(class_weight='balanced', probability=True, random_state=42),
        'params': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
        'needs_scaling': True
    },
    {
        'name': 'XGBoost',
        'model': XGBClassifier(eval_metric='logloss', random_state=42),
        'params': {'n_estimators': [100, 200], 'max_depth': [3, 5, 10], 'learning_rate': [0.01, 0.1]},
        'needs_scaling': False
    },
]

results = []

print("\n" + "=" * 70)
print("COMPARAISON DES ALGORITHMES")
print("=" * 70)

for exp in experiments:
    print(f"\n--- {exp['name']} ---")

    with mlflow.start_run(run_name=exp['name']):
        X_tr = X_train_scaled if exp['needs_scaling'] else X_train
        X_te = X_test_scaled if exp['needs_scaling'] else X_test

        grid = GridSearchCV(
            exp['model'], exp['params'],
            cv=5, scoring='f1', n_jobs=-1, verbose=0
        )
        grid.fit(X_tr, y_train)
        best = grid.best_estimator_

        preds = best.predict(X_te)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        mlflow.log_param("algorithm", exp['name'])
        mlflow.log_param("best_params", str(grid.best_params_))
        mlflow.log_param("needs_scaling", exp['needs_scaling'])
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("best_cv_f1", grid.best_score_)
        mlflow.sklearn.log_model(best, "model")

        results.append({
            'algorithm': exp['name'],
            'best_params': grid.best_params_,
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1,
            'cv_f1': grid.best_score_, 'needs_scaling': exp['needs_scaling']
        })

        print(f"  Best params : {grid.best_params_}")
        print(f"  F1-Score    : {f1:.4f} | Accuracy : {acc:.4f}")

# SÉLECTION DU MEILLEUR MODÈLE
results_df = pd.DataFrame(results).sort_values('f1_score', ascending=False)
best_result = results_df.iloc[0]

print("\n" + "=" * 70)
print("RÉSULTATS COMPARATIFS")
print("=" * 70)
print(results_df[['algorithm', 'accuracy', 'precision', 'recall', 'f1_score', 'cv_f1']].to_string(index=False))
print(f"\n🏆 Meilleur algorithme : {best_result['algorithm']} (F1={best_result['f1_score']:.4f})")

# RÉENTRAÎNEMENT FINAL & SAUVEGARDE
best_exp = next(e for e in experiments if e['name'] == best_result['algorithm'])
needs_scaling = best_exp['needs_scaling']
X_tr_final = X_train_scaled if needs_scaling else X_train

final_model = best_exp['model'].set_params(**best_result['best_params'])
final_model.fit(X_tr_final, y_train)
joblib.dump(final_model, MODEL_PATH)

# On ne sauvegarde le scaler QUE si le modèle retenu en a besoin
if needs_scaling:
    joblib.dump(scaler, SCALER_PATH)
elif os.path.exists(SCALER_PATH):
    os.remove(SCALER_PATH)

metrics = {
    "best_algorithm": best_result['algorithm'],
    "accuracy": float(best_result['accuracy']),
    "precision": float(best_result['precision']),
    "recall": float(best_result['recall']),
    "f1_score": float(best_result['f1_score']),
    "best_params": str(best_result['best_params']),
    "needs_scaling": bool(needs_scaling),
    "dataset": "diabetes",
    "threshold": SEUIL_ACCEPTATION
}
with open(METRICS_PATH, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n✅ Modèle sauvegardé : {MODEL_PATH}")
print(f"Métriques sauvegardées : {METRICS_PATH}")

if best_result['f1_score'] >= SEUIL_ACCEPTATION:
    print(f"✅ MODÈLE VALIDÉ (F1 {best_result['f1_score']:.4f} >= {SEUIL_ACCEPTATION})")
else:
    print(f"❌ MODÈLE REJETÉ (F1 {best_result['f1_score']:.4f} < {SEUIL_ACCEPTATION})")