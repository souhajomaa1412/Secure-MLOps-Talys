# 🩺 Secure MLOps Pipeline for Diabetes Prediction

## 📌 Description

Ce projet consiste à concevoir et implémenter un **pipeline MLOps sécurisé** pour la prédiction du diabète.

Le pipeline couvre le cycle de vie du modèle :

* Préparation et versioning des données
* Entraînement et comparaison des modèles
* Tracking avec **MLflow**
* Orchestration avec **Apache Airflow**
* API de prédiction avec **FastAPI**
* Conteneurisation avec **Docker**
* Déploiement avec **Kubernetes**
* Monitoring avec **Prometheus et Grafana**
* Sécurité Kubernetes avec **Secrets, ServiceAccount, NetworkPolicy et TLS**
* PoC: Inférence sécurisée avec **FHE**

---

## 🏗️ Architecture

```text
                         Dataset
                            │
                            ▼
                     ┌─────────────┐
                     │   Airflow   │
                     └──────┬──────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │ Training + MLflow│
                  └────────┬─────────┘
                           │
                           ▼
                     ┌───────────┐
                     │  FastAPI  │
                     └─────┬─────┘
                           │
                    ┌──────┴──────┐
                    ▼             ▼
               PostgreSQL        FHE
                               Inference
                           
                     Monitoring
                           │
                    ┌──────┴──────┐
                    ▼             ▼
               Prometheus       Grafana

                    Kubernetes
```

---

## 🛠️ Technologies

* **Python**
* **FastAPI**
* **Scikit-learn**
* **XGBoost**
* **MLflow**
* **Apache Airflow**
* **PostgreSQL**
* **Docker**
* **Docker Compose**
* **Kubernetes**
* **Prometheus**
* **Grafana**
* **FHE**
* **DVC**
* **Git / GitHub**
---
# 📁 Structure du projet

```text
projet-pipeline/
│
├── dags/
│   └── Airflow DAGs
│
├── data/
│   └── Datasets
│
├── models/
│   └── Modèles entraînés
│
├── src/
│   └── Code source
│
├── monitoring/
│   └── prometheus.yml
│
├── k8s/
│   ├── namespace.yml
│   ├── api.yml
│   ├── postgres.yml
│   ├── postgres-secret.yaml
│   ├── api-secrets.yaml
│   ├── rbac.yaml
│   ├── network-security.yaml
│   └── tls-selfsigned.yaml
│
├── initdb/
│   └── init.sql
│
├── Dockerfile
├── Dockerfile.airflow
├── docker-compose.yml
├── docker-compose.monitoring.yml
├── requirements.txt
```
---


# 🚀 Installation

## 1. Cloner le projet

```bash
git clone https://github.com/souhajomaa1412/mlops-pipeline.git
cd mlops-pipeline
```

## 2. Lancer les services

```bash
docker compose up -d
```

Vérifier les conteneurs :

```bash
docker compose ps
```

Arrêter les services :

```bash
docker compose down
```

---

# 🌐 Interfaces

Après le démarrage :

| Service         | Adresse                    |
| --------------- | -------------------------- |
| FastAPI         | http://localhost:8000      |
| FastAPI Swagger | http://localhost:8000/docs |
| MLflow          | http://localhost:5000      |
| Airflow         | http://localhost:8092      |
| Prometheus      | http://localhost:9090      |
| Grafana         | http://localhost:3000      |

---

# 📊 Monitoring

Le monitoring est réalisé avec **Prometheus et Grafana**.

Prometheus collecte les métriques de l'API et du GPU.

Grafana permet de visualiser notamment :

* Nombre de requêtes
* Temps de réponse
* Taux d'erreur
* Utilisation du GPU
* Métriques de performance

---

# ☸️ Kubernetes

Les fichiers Kubernetes se trouvent dans le dossier :

```text
k8s/
├── namespace.yml
├── api.yml
├── postgres.yml
├── postgres-secret.yaml
├── api-secrets.yaml
├── rbac.yaml
├── network-security.yaml
└── tls-selfsigned.yaml
```

## 1. Créer le namespace

```bash
kubectl apply -f k8s/namespace.yml
```

## 2. Déployer PostgreSQL

```bash
kubectl apply -f k8s/postgres-secret.yaml
kubectl apply -f k8s/postgres.yml
```

## 3. Déployer l'API

```bash
kubectl apply -f k8s/api-secrets.yaml
kubectl apply -f k8s/api.yml
```

Vérifier les pods :

```bash
kubectl get pods -n mlops
```

Vérifier les services :

```bash
kubectl get services -n mlops
```

---

# 🔒 TLS et Ingress

Le projet utilise **Ingress NGINX** et **cert-manager** pour sécuriser l'accès à l'API avec TLS.

```bash
kubectl apply -f k8s/tls-selfsigned.yaml
```

Vérifier le certificat :

```bash
kubectl get certificate -n mlops
```

Vérifier l'Ingress :

```bash
kubectl get ingress -n mlops
```

---

# 🛡️ Kubernetes Security

La sécurité Kubernetes utilise :

```text
API
 │
 ▼
ServiceAccount
 │
 ▼
Kubernetes
 │
 ├── Secrets
 ├── NetworkPolicy
 └── TLS / Ingress
```

Les communications vers PostgreSQL sont limitées grâce à la **NetworkPolicy**.

---

# 🔐 FHE Inference

Le projet utilise **Fully Homomorphic Encryption (FHE)** afin de permettre l'inférence sur des données chiffrées.

L'objectif est de protéger les données sensibles pendant le processus de prédiction.

Cette sécurité entraîne cependant un **coût en performance** par rapport à l'inférence en clair.

---

# 📦 Docker Image

L'image de l'API est disponible sur **GitHub Container Registry (GHCR)**.

Pour récupérer l'image :

```bash
docker pull ghcr.io/souhajomaa1412/mlops-pipeline:a0ae79038091ef3d1d6d7d5b34e3eaa331e6ad0e
```

Vérifier l'image :

```bash
docker images
```

---


---

# 🧪 Résultats

Le pipeline permet :

* ✅ L'automatisation de l'entraînement
* ✅ Le suivi des expériences ML
* ✅ Le déploiement d'une API de prédiction
* ✅ La persistance des prédictions
* ✅ Le monitoring de l'application
* ✅ Le monitoring GPU
* ✅ Le déploiement Kubernetes
* ✅ La sécurisation des communications avec TLS
* ✅ La protection des données avec FHE

---



# 👩‍💻 Auteur

**Souha Jomaa**

Computer Engineering Student
**Cloud Computing & Cybersecurity**

---
