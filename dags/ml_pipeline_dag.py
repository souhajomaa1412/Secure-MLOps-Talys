# noinspection PyUnresolvedReferences
from airflow import DAG # noqa
# noinspection PyUnresolvedReferences
from airflow.operators.python import PythonOperator # noqa
from datetime import datetime, timedelta
import subprocess
import sys
import os

default_args = {
    'owner': 'souhajo',
    'depends_on_past': False,
    'start_date': datetime(2026, 7, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'mlops_diabetes_pipeline',
    default_args=default_args,
    description='Pipeline MLOps - Diabetes Prediction',
    schedule='@daily',
    catchup=False
)

def extract_data():
    import pandas as pd # noqa
    df = pd.read_csv('/opt/airflow/data/diabetes.csv')
    print(f"Data extracted: {df.shape}")
    return df.shape


def preprocess_data():
    import subprocess
    result = subprocess.run(
        ['python', '/opt/airflow/src/preprocess_spark.py'],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise Exception(result.stderr)

def train_model():
    import subprocess
    result = subprocess.run(
        ['python', '/opt/airflow/src/train.py'],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise Exception(result.stderr)

def validate_model():
    import subprocess
    result = subprocess.run(
        ['python', '/opt/airflow/src/validate.py'],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise Exception("Model validation failed!")


def security_check():

    from cryptography.fernet import Fernet, InvalidToken

    key = os.environ.get("ENCRYPTION_KEY")
    if not key:
        raise Exception(
            "SECURITY CHECK FAILED: ENCRYPTION_KEY absente. "
            "Le pipeline ne doit pas être considéré prêt pour un déploiement "
            "sans clé de chiffrement configurée."
        )

    try:
        fernet = Fernet(key.encode())
        test_payload = b'{"test": "security_check"}'
        encrypted = fernet.encrypt(test_payload)
        decrypted = fernet.decrypt(encrypted)
        assert decrypted == test_payload
        print("✅ SECURITY CHECK PASSED : chiffrement/déchiffrement fonctionnel")
    except (InvalidToken, ValueError, AssertionError) as e:
        raise Exception(f"SECURITY CHECK FAILED : clé invalide ou aller-retour échoué ({e})")


# Définir les tâches
task_extract = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag
)

task_preprocess = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag
)

task_train = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

task_validate = PythonOperator(
    task_id='validate_model',
    python_callable=validate_model,
    dag=dag
)

task_security = PythonOperator(
    task_id='security_check',
    python_callable=security_check,
    dag=dag
)


task_extract >> task_preprocess >> task_train >> task_validate >> task_security
