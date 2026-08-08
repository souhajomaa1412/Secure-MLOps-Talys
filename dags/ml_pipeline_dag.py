# noinspection PyUnresolvedReferences
from airflow import DAG # noqa
# noinspection PyUnresolvedReferences
from airflow.operators.python import PythonOperator # noqa
from datetime import datetime, timedelta
import subprocess
import sys

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

# Ordre d'exécution du DAG task_extract >> task_preprocess >> task_train >> task_validate