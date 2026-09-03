# 🩺 Secure MLOps Pipeline for Diabetes Prediction

## 📌 Description

This project consists of designing and implementing a **secure MLOps pipeline for diabetes prediction**.

The pipeline covers the main stages of the machine learning lifecycle:

- Data preparation and versioning
- Model training and comparison
- Experiment tracking with **MLflow**
- Workflow orchestration with **Apache Airflow**
- Prediction API with **FastAPI**
- Containerization with **Docker**
- CI/CD with **GitHub Actions**
- Deployment with **Kubernetes**
- Monitoring with **Prometheus and Grafana**
- Kubernetes security with **Secrets, ServiceAccount, NetworkPolicy and TLS**
- Proof of Concept for secure inference using **Fully Homomorphic Encryption (FHE)**

---

## 🏗️ Architecture

<img width="1512" height="322" alt="Pipeline-dark drawio" src="https://github.com/user-attachments/assets/011cc24b-7c76-408b-83eb-404ffb5a593f" />

---
🛠️ Technologies
Machine Learning
Python
Scikit-learn
XGBoost
MLOps
MLflow
Apache Airflow
DVC
GitHub Actions
API and Backend
FastAPI
PostgreSQL
Containerization and Deployment
Docker
Docker Compose
Kubernetes
Monitoring
Prometheus
Grafana
Security
Kubernetes Secrets
ServiceAccount
NetworkPolicy
TLS / HTTPS
Ingress NGINX
FHE (Fully Homomorphic Encryption) PoC
📁 Project Structure
projet-pipeline/
│
├── dags/
│   └── Airflow DAGs
│
├── data/
│   └── Datasets
│
├── models/
│   └── Trained models
│
├── src/
│   └── Source code
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
└── README.md
🚀 Installation
1. Clone the repository
git clone https://github.com/souhajomaa1412/mlops-pipeline.git
cd mlops-pipeline
2. Start the MLOps services
docker compose up -d

Check the running containers:

docker compose ps

Stop the services:

docker compose down
🌐 Application Interfaces

After starting the services, the main interfaces are available at:

Service	Address
FastAPI	http://localhost:8000
FastAPI Swagger	http://localhost:8000/docs
MLflow	http://localhost:5000
Airflow	http://localhost:8092
Prometheus	http://localhost:9090
Grafana	http://localhost:3000
🤖 Machine Learning

The project trains and compares several machine learning algorithms for diabetes prediction.

The evaluated models include:

Logistic Regression
Random Forest
SVM with RBF kernel
XGBoost

The best model is selected according to the evaluation metrics and can then be used by the prediction API.

📊 MLflow

MLflow is used to track machine learning experiments.

It allows the project to track:

Model parameters
Evaluation metrics
Training runs
Model artifacts
Best model

MLflow provides a centralized interface for comparing different training experiments.

🔄 Apache Airflow

Apache Airflow is used to orchestrate the machine learning workflow.

The workflow can automate tasks such as:

Data
  ↓
Training
  ↓
Validation
  ↓
Model Selection
  ↓
Deployment

This allows the ML pipeline to be executed in an automated and reproducible way.

🧪 Testing and CI/CD

Automated tests are implemented using pytest.

The CI/CD workflow verifies the project before deployment.

Main steps include:

Push to GitHub
      ↓
Install dependencies
      ↓
Run tests
      ↓
Validate the model
      ↓
Build Docker image
      ↓
Push image to GHCR

The raw dataset is included in the project so that the CI workflow can execute and validate the pipeline automatically.

🐳 Docker

The FastAPI application is containerized using Docker.

Build the image:

docker build -t mlops-pipeline .

Run the container:

docker run -p 8000:8000 mlops-pipeline
📦 Docker Image from GHCR

The project image is also available through GitHub Container Registry (GHCR).

Pull the image:

docker pull ghcr.io/souhajomaa1412/mlops-pipeline:a0ae79038091ef3d1d6d7d5b34e3eaa331e6ad0e

Check the downloaded image:

docker images
☸️ Kubernetes Deployment

The Kubernetes configuration is located in:

k8s/
1. Create the namespace
kubectl apply -f k8s/namespace.yml
2. Deploy PostgreSQL
kubectl apply -f k8s/postgres-secret.yaml
kubectl apply -f k8s/postgres.yml
3. Deploy the API
kubectl apply -f k8s/api-secrets.yaml
kubectl apply -f k8s/api.yml

Check the pods:

kubectl get pods -n mlops

Check the services:

kubectl get services -n mlops
🔐 Kubernetes Security

The Kubernetes deployment includes several security mechanisms.

Secrets

Sensitive configuration such as database credentials and API configuration is stored using Kubernetes Secrets.

api-secrets.yaml
postgres-secret.yaml
ServiceAccount

The FastAPI application runs with a dedicated Kubernetes ServiceAccount:

mlops-api-sa

The ServiceAccount token is not automatically mounted into the pod.

NetworkPolicy

A NetworkPolicy restricts access to PostgreSQL.

Only the FastAPI pods are allowed to communicate with PostgreSQL on port 5432.

FastAPI Pods
     │
     │ TCP 5432
     ▼
 PostgreSQL

This reduces unnecessary network access between Kubernetes workloads.

🔒 TLS and Ingress

The API is exposed through an NGINX Ingress with TLS.

The project uses cert-manager to generate and manage the TLS certificate.

Apply the TLS and Ingress configuration:

kubectl apply -f k8s/tls-selfsigned.yaml

Check the certificate:

kubectl get certificate -n mlops

Check the Ingress:

kubectl get ingress -n mlops

The API is intended to be accessed through:

https://mlops-api.local
📈 Monitoring

The application is monitored using Prometheus and Grafana.

The monitoring system collects application and infrastructure metrics.

Examples include:

Number of API requests
API response time
HTTP error rate
GPU utilization
Application performance

Architecture:

FastAPI
   │
   ▼
Prometheus
   │
   ▼
Grafana

Grafana dashboards provide a visual overview of the system performance.

🔐 FHE Inference PoC

The project also includes a Proof of Concept using Fully Homomorphic Encryption (FHE).

The objective is to perform machine learning inference while keeping input data encrypted.

Sensitive Data
      ↓
   Encryption
      ↓
Encrypted Data
      ↓
 FHE Inference
      ↓
Prediction

FHE improves data privacy but introduces an additional computational cost compared with standard plaintext inference.

📊 Results

The pipeline provides:

✅ Automated model training
✅ Comparison of multiple ML models
✅ Experiment tracking with MLflow
✅ Reproducible data management with DVC
✅ Automated testing and CI/CD
✅ FastAPI prediction service
✅ Docker containerization
✅ Kubernetes deployment
✅ PostgreSQL persistence
✅ Prometheus and Grafana monitoring
✅ Kubernetes Secrets
✅ Dedicated ServiceAccount
✅ NetworkPolicy for database protection
✅ HTTPS with TLS
✅ FHE inference Proof of Concept
🔒 Security Overview



The project therefore combines MLOps automation, cloud-native deployment, monitoring, and security mechanisms.

👩‍💻 Author

Souha Jomaa

Computer Engineering Student
Cloud Computing & Cybersecurity
---
