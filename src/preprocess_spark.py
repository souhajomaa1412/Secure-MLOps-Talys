from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql.types import DoubleType
import pandas as pd

def preprocess_with_spark(input_path='data/diabetes.csv',
                           output_path='data/diabetes_processed.csv'):

    # Créer session Spark
    spark = SparkSession.builder \
        .appName("DiabetesPreprocessing") \
        .master("local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    print("Chargement des données avec Spark...")
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    print(f"Shape initiale : {df.count()} lignes × {len(df.columns)} colonnes")
    df.show(5)

    # ── 1. REMPLACER LES ZÉROS ABERRANTS ──
    # Dans ce dataset, 0 = valeur manquante pour ces colonnes
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI']

    for col_name in zero_cols:
        mean_val = df.filter(col(col_name) > 0) \
                     .agg(mean(col_name)).collect()[0][0]
        df = df.withColumn(
            col_name,
            when(col(col_name) == 0, mean_val).otherwise(col(col_name))
        )
        print(f"Colonne {col_name} : zéros remplacés par moyenne ({mean_val:.2f})")

    # ── 2. STATISTIQUES APRÈS NETTOYAGE ──
    print("\nStatistiques après nettoyage :")
    df.describe().show()

    # ── 3. SAUVEGARDER EN CSV POUR TRAIN.PY ──
    pandas_df = df.toPandas()
    pandas_df.to_csv(output_path, index=False)
    print(f"\nDonnées préprocessées sauvegardées : {output_path}")
    print(f"Shape finale : {pandas_df.shape}")

    spark.stop()
    return output_path


if __name__ == "__main__":
    preprocess_with_spark()