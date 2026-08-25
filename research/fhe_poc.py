"""
POC FHE (Fully Homomorphic Encryption) — preuve de concept, hors pipeline
de production.
"""

import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from concrete.ml.sklearn import LogisticRegression as FHELogisticRegression
from sklearn.linear_model import LogisticRegression as ClearLogisticRegression

DATASET_PATH = "data/diabetes_processed.csv"

# DONNÉES
df = pd.read_csv(DATASET_PATH)

X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# MODÈLE EN CLAIR
print("Entraînement du modèle en clair...")
clear_model = ClearLogisticRegression(max_iter=1000)
clear_model.fit(X_train, y_train)

# MODÈLE FHE

print("Entraînement du modèle FHE...")

fhe_model = FHELogisticRegression(n_bits=8)
fhe_model.fit(X_train, y_train)

print("Compilation du circuit FHE...")
fhe_model.compile(X_train)

# COMPARAISON SUR  ÉCHANTILLONS

n_test = min(20, len(X_test))

clear_predictions = []
fhe_predictions = []

print("\nExécution des prédictions FHE...")
print("=" * 60)

total_fhe_time = 0

for i in range(n_test):

    sample = X_test[i:i + 1]

    # ----- Prédiction en clair -----
    pred_clear = clear_model.predict(sample)[0]

    # ----- Prédiction FHE -----
    t0 = time.perf_counter()

    pred_fhe = fhe_model.predict(
        sample,
        fhe="execute"
    )[0]

    t_fhe = time.perf_counter() - t0

    total_fhe_time += t_fhe

    clear_predictions.append(pred_clear)
    fhe_predictions.append(pred_fhe)

    status = "✓" if pred_clear == pred_fhe else "✗ DIVERGENCE"

    print(
        f"Échantillon {i:02d}: "
        f"clair={pred_clear} | "
        f"FHE={pred_fhe} | "
        f"{t_fhe * 1000:.2f} ms | "
        f"{status}"
    )


# RÉSULTATS

agreement = sum(
    clear_predictions[i] == fhe_predictions[i]
    for i in range(n_test)
)

agreement_rate = agreement / n_test

clear_accuracy = accuracy_score(
    y_test[:n_test],
    clear_predictions
)

fhe_accuracy = accuracy_score(
    y_test[:n_test],
    fhe_predictions
)

clear_f1 = f1_score(
    y_test[:n_test],
    clear_predictions,
    zero_division=0
)

fhe_f1 = f1_score(
    y_test[:n_test],
    fhe_predictions,
    zero_division=0
)

average_fhe_time = total_fhe_time / n_test

print("\n" + "=" * 60)
print("RÉSULTATS POC FHE")
print("=" * 60)

print(f"Nombre d'échantillons       : {n_test}")

print(
    f"Taux d'accord clair/FHE    : "
    f"{agreement}/{n_test} ({agreement_rate * 100:.1f}%)"
)

print(f"Accuracy modèle clair       : {clear_accuracy * 100:.2f}%")
print(f"Accuracy modèle FHE         : {fhe_accuracy * 100:.2f}%")

print(f"F1-score modèle clair       : {clear_f1:.4f}")
print(f"F1-score modèle FHE         : {fhe_f1:.4f}")

print(
    f"Temps moyen inférence FHE   : "
    f"{average_fhe_time * 1000:.2f} ms"
)

print("=" * 60)

print("\nConclusion :")

if agreement_rate == 1:
    print(
        "Le modèle FHE produit les mêmes prédictions que "
        "le modèle en clair sur les échantillons testés."
    )
else:
    print(
        f"Le modèle FHE est cohérent avec le modèle en clair "
        f"sur {agreement_rate * 100:.1f}% des échantillons."
    )
    print(
        "Les divergences peuvent être liées à la quantification "
        "utilisée par le modèle FHE."
    )

print(
    "Le temps d'inférence FHE est supérieur à celui d'une "
    "inférence classique, ce qui constitue un compromis "
    "entre confidentialité et performance."
)