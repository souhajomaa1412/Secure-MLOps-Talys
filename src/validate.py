import json
import sys

METRICS_PATH = 'metrics.json'

SEUIL_ACCEPTATION = 0.65
def validate_model():
    print("=" * 50)
    print("Validation du modèle - Diabetes")
    print("=" * 50)

    try:
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
    except FileNotFoundError:
        print("❌ ERREUR : metrics.json introuvable !")
        sys.exit(1)

    f1 = metrics.get('f1_score', 0)
    seuil = metrics.get('threshold', 0.75)
    algo = metrics.get('best_algorithm', 'unknown')

    print(f"\nAlgorithme : {algo}")
    print(f"Accuracy   : {metrics.get('accuracy', 0) * 100:.2f}%")
    print(f"Precision  : {metrics.get('precision', 0) * 100:.2f}%")
    print(f"Recall     : {metrics.get('recall', 0) * 100:.2f}%")
    print(f"F1-Score   : {f1 * 100:.2f}%")
    print(f"Seuil F1   : {seuil * 100:.0f}%")
    print("-" * 50)

    if f1 >= seuil:
        print("✅ MODÈLE VALIDÉ")
        return True
    else:
        print("❌ MODÈLE REJETÉ — ancien modèle conservé")
        return False


if __name__ == "__main__":
    sys.exit(0 if validate_model() else 1)