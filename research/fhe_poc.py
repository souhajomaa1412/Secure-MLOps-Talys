"""
POC FHE (Fully Homomorphic Encryption) — preuve de concept,
hors pipeline de production.

Objectifs :
- comparer les prédictions en clair et en FHE ;
- mesurer le taux d'accord entre les deux modèles ;
- mesurer les performances d'inférence ;
- évaluer le compromis entre confidentialité et performance.
"""

import time

import pandas as pd
from sklearn.linear_model import LogisticRegression as ClearLogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from concrete.ml.sklearn import LogisticRegression as FHELogisticRegression


DATASET_PATH = "data/diabetes_processed.csv"

# Configuration de la POC
N_BITS = 12
N_SAMPLES = 100


# ============================================================
# DONNÉES
# ============================================================

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


# ============================================================
# MODÈLE EN CLAIR
# ============================================================

print("Entraînement du modèle en clair...")

clear_model = ClearLogisticRegression(max_iter=1000)
clear_model.fit(X_train, y_train)


# ============================================================
# MODÈLE FHE
# ============================================================

print("Entraînement du modèle FHE...")

fhe_model = FHELogisticRegression(n_bits=N_BITS)
fhe_model.fit(X_train, y_train)

print("Compilation du circuit FHE...")

fhe_model.compile(X_train)


# ============================================================
# COMPARAISON SUR LES ÉCHANTILLONS
# ============================================================

n_test = min(N_SAMPLES, len(X_test))

clear_predictions = []
fhe_predictions = []

total_clear_time = 0
total_fhe_time = 0

print("\nExécution des prédictions FHE...")
print("=" * 90)

for i in range(n_test):

    sample = X_test[i:i + 1]

    # --------------------------------------------------------
    # Prédiction en clair
    # --------------------------------------------------------

    t0 = time.perf_counter()

    pred_clear = clear_model.predict(sample)[0]

    t_clear = time.perf_counter() - t0
    total_clear_time += t_clear

    # --------------------------------------------------------
    # Prédiction FHE
    # --------------------------------------------------------

    t0 = time.perf_counter()

    pred_fhe = fhe_model.predict(
        sample,
        fhe="execute"
    )[0]

    t_fhe = time.perf_counter() - t0
    total_fhe_time += t_fhe

    # --------------------------------------------------------
    # Stockage des résultats
    # --------------------------------------------------------

    clear_predictions.append(pred_clear)
    fhe_predictions.append(pred_fhe)

    if pred_clear == pred_fhe:
        status = "✓"
    else:
        status = "✗ DIVERGENCE"

    print(
        f"Échantillon {i:03d}: "
        f"clair={pred_clear} | "
        f"FHE={pred_fhe} | "
        f"clair={t_clear * 1000:.4f} ms | "
        f"FHE={t_fhe * 1000:.2f} ms | "
        f"{status}"
    )


# ============================================================
# MÉTRIQUES
# ============================================================

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


# ============================================================
# PERFORMANCES
# ============================================================

average_clear_time = total_clear_time / n_test
average_fhe_time = total_fhe_time / n_test

if average_clear_time > 0:
    fhe_overhead = average_fhe_time / average_clear_time
else:
    fhe_overhead = 0


# ============================================================
# RÉSULTATS
# ============================================================

print("\n" + "=" * 70)
print("RÉSULTATS POC FHE")
print("=" * 70)

print(f"Nombre d'échantillons       : {n_test}")
print(f"Nombre de bits (n_bits)     : {N_BITS}")

print(
    f"Taux d'accord clair/FHE    : "
    f"{agreement}/{n_test} "
    f"({agreement_rate * 100:.1f}%)"
)

print()

print(
    f"Accuracy modèle clair       : "
    f"{clear_accuracy * 100:.2f}%"
)

print(
    f"Accuracy modèle FHE         : "
    f"{fhe_accuracy * 100:.2f}%"
)

print()

print(
    f"F1-score modèle clair       : "
    f"{clear_f1:.4f}"
)

print(
    f"F1-score modèle FHE         : "
    f"{fhe_f1:.4f}"
)

print()

print(
    f"Temps moyen inférence clair : "
    f"{average_clear_time * 1000:.4f} ms"
)

print(
    f"Temps moyen inférence FHE   : "
    f"{average_fhe_time * 1000:.2f} ms"
)

print(
    f"Surcoût FHE                 : "
    f"x{fhe_overhead:.1f}"
)

print("=" * 70)


# ============================================================
# CONCLUSION
# ============================================================

print("\nConclusion :")

print(
    f"Le modèle FHE produit les mêmes prédictions que le modèle "
    f"en clair dans {agreement_rate * 100:.1f}% des cas "
    f"({agreement}/{n_test})."
)

if agreement_rate < 1:
    divergences = n_test - agreement

    print(
        f"{divergences} divergence(s) ont été observées. "
        f"Ces différences peuvent notamment être liées à la "
        f"quantification utilisée pour l'inférence FHE."
    )

print(
    f"Le temps moyen d'inférence est de "
    f"{average_clear_time * 1000:.4f} ms en clair contre "
    f"{average_fhe_time * 1000:.2f} ms en FHE."
)

print(
    f"Le surcoût mesuré de l'inférence FHE est d'environ "
    f"x{fhe_overhead:.1f}."
)


# si nb_Échantillon=20 n_bits=6 : taux d'accord clair/FHE    : 17/20 (85.0%), 3 DIVERGENCE

# si nb_Échantillon=20 n_bits=8 : taux d'accord clair/FHE    : 18/20 (90.0%), 2 DIVERGENCE

# si nb_Échantillon=100 n_bits=12 : taux d'accord clair/FHE    : 95/100 (95.0%), 5 DIVERGENCE

