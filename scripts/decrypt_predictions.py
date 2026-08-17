"""
Script d'audit : déchiffre les prédictions stockées en base, pour un usage
exemple
"""
import os
import json
import psycopg2
from cryptography.fernet import Fernet, InvalidToken
from dotenv import load_dotenv #.env
load_dotenv()
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/mlops_predictions"
)
ENCRYPTION_KEY = os.environ.get("ENCRYPTION_KEY")

if not ENCRYPTION_KEY:
    raise SystemExit("ENCRYPTION_KEY manquante dans l'environnement.")

fernet = Fernet(ENCRYPTION_KEY.encode())

conn = psycopg2.connect(DATABASE_URL)
with conn.cursor() as cur:
    cur.execute(
        "SELECT id, created_at, encrypted_data, prediction, confidence, algorithm "
        "FROM predictions ORDER BY created_at DESC LIMIT 20;"
    )
    rows = cur.fetchall()

for row_id, created_at, encrypted_data, prediction, confidence, algorithm in rows:
    try:
        clinical = json.loads(fernet.decrypt(bytes(encrypted_data)).decode("utf-8"))
        print(f"[{row_id}] {created_at} | pred={prediction} conf={confidence} algo={algorithm}")
        print(f"    -> {clinical}")
    except InvalidToken:
        print(f"[{row_id}] {created_at} — déchiffrement impossible (mauvaise clé ?)")

conn.close()