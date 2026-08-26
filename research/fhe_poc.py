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
from concrete.ml.sklearn import LogisticRegression as FHELogisticRegression
from sklearn.linear_model import LogisticRegression as ClearLogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "data/diabetes_processed.csv"

N_BITS = 12
N_SAMPLES = 100


# ============================================================
# DONNÉES
# ============================================================

print("Chargement des données...")

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
# STANDARDISATION
# ============================================================
# Même preprocessing que celui utilisé pour la régression
# logistique dans le pipeline de production.

print("Standardisation des données...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# MODÈLE EN CLAIR
# ============================================================

print("Entraînement du modèle en clair...")

clear_model = ClearLogisticRegression(
    max_iter=1000,
    random_state=42
)

clear_model.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# MODÈLE FHE
# ============================================================

print("Entraînement du modèle FHE...")

fhe_model = FHELogisticRegression(
    n_bits=N_BITS
)

fhe_model.fit(
    X_train_scaled,
    y_train
)

print("Compilation du circuit FHE...")

fhe_model.compile(
    X_train_scaled
)


# ============================================================
# COMPARAISON
# ============================================================

n_test = min(N_SAMPLES, len(X_test_scaled))

clear_predictions = []
fhe_predictions = []

total_clear_time = 0.0
total_fhe_time = 0.0

print("\nExécution des prédictions FHE...")
print("=" * 90)


for i in range(n_test):

    # Un seul échantillon standardisé
    sample = X_test_scaled[i:i + 1]

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
    # Stockage
    # --------------------------------------------------------

    clear_predictions.append(pred_clear)
    fhe_predictions.append(pred_fhe)

    # --------------------------------------------------------
    # Comparaison
    # --------------------------------------------------------

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
# TAUX D'ACCORD
# ============================================================

agreement = sum(
    clear_predictions[i] == fhe_predictions[i]
    for i in range(n_test)
)

agreement_rate = agreement / n_test


# ============================================================
# ACCURACY
# ============================================================

clear_accuracy = accuracy_score(
    y_test[:n_test],
    clear_predictions
)

fhe_accuracy = accuracy_score(
    y_test[:n_test],
    fhe_predictions
)


# ============================================================
# F1-SCORE
# ============================================================

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

print(
    f"Nombre d'échantillons       : {n_test}"
)

print(
    f"Nombre de bits (n_bits)     : {N_BITS}"
)

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
        f"{divergences} divergence(s) ont été observées."
    )

    print(
        "Ces différences peuvent notamment être liées à la "
        "quantification utilisée lors de la compilation FHE."
    )

else:

    print(
        "Aucune divergence n'a été observée sur les échantillons testés."
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

print(
    "La POC montre ainsi le compromis entre la confidentialité "
    "des données et les performances d'inférence."
)
"""
La POC montre que, avec n_bits=12,
les prédictions FHE sont identiques aux prédictions du modèle en clair sur les 100 échantillons testés. 
Aucune divergence n'a été observée.'
 En contrepartie, l'inférence FHE est environ 40 fois plus lente que l'inférence en clair (21,00 ms contre 0,5242 ms). 
Ces résultats illustrent le compromis entre confidentialité des données et performances d'inférence."""