# ChurnAI — Pipeline MLOps Complet | Groupe 1

> **MLOps II — SupNum DEML M2 | Professeur Yehdhih ANNA**

Prédiction de la résiliation client (churn) sur le dataset Telco Customer Churn, avec un pipeline MLOps complet de la donnée brute au modèle en production.

---

## Membres du Groupe

| Matricule | Rôle | Branche |
|-----------|------|---------|
| 21003 | ML Engineer — Pipeline DVC, MLflow, entraînement | `dev-21003` |
| 21008 | Backend Dev — API FastAPI, Dockerfile, endpoint `/predict` | `dev-21008` |
| 21027 | DevOps / Cloud — EC2, S3, CI/CD GitHub Actions | `dev-21027` |
| 21032 | Frontend Dev — Interface utilisateur, export CSV, README | `dev-21032` |

---

## Services en Production

| Service | URL |
|---------|-----|
| Interface Web | http://15.224.61.131:8000/ui |
| API REST | http://15.224.61.131:8000 |
| Health Check | http://15.224.61.131:8000/ |
| MLflow UI | http://15.224.61.131:5000 |

---

## Architecture

```
GitHub Repository (main / staging / dev-MATRICULE)
        │
        ├── merge → main ──────────────────────────────────────────┐
        │                                                           ▼
        └── .dvc modifié ──────────────────────────────> GitHub Actions
                                                         ┌─────────────────────────┐
                                                         │ Workflow 1 (Code CI)    │
                                                         │  → SSH → EC2            │
                                                         │  → docker rebuild       │
                                                         │  → restart API          │
                                                         │  → health check         │
                                                         │                         │
                                                         │ Workflow 2 (Data CI)    │
                                                         │  → SSH → EC2            │
                                                         │  → dvc repro            │
                                                         │  → MLflow log           │
                                                         │  → restart API          │
                                                         └──────────┬──────────────┘
                                                                    │ SSH
                                                         ┌──────────▼──────────────┐
                                                         │   AWS EC2 (t3.medium)   │
                                                         │                         │
                                                         │  MLflow :5000           │
                                                         │  Docker API :8000       │
                                                         └──────────┬──────────────┘
                                                                    │
                                                         ┌──────────▼──────────────┐
                                                         │     AWS S3 Bucket       │
                                                         │  data/raw/              │
                                                         │  data/processed/        │
                                                         │  MLflow artifacts cache │
                                                         │  DVC remote store       │
                                                         └─────────────────────────┘
```

---

## Pipeline DVC

Le pipeline ML est structuré en 4 stages exécutables avec `dvc repro` :

| Stage | Script | Description |
|-------|--------|-------------|
| `preprocess` | `src/preprocess.py` | Chargement depuis S3, nettoyage, encodage, normalisation, split train/test |
| `train` | `src/train.py` | Entraînement Random Forest, logging MLflow |
| `evaluate` | `src/evaluate.py` | Métriques sur jeu de test (accuracy, F1, ROC-AUC) |
| `register` | `src/register.py` | Enregistrement conditionnel dans MLflow Registry si accuracy ≥ seuil |

**Modèle :** Random Forest | **Accuracy :** 79.6% | **ROC-AUC :** 83.6%

---

## API REST

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Health check — retourne le statut et le modèle actif |
| `/ui` | GET | Interface web |
| `/predict` | POST | Prédiction à partir des 19 features client |
| `/reload` | POST | Rechargement du modèle depuis MLflow Registry |

**Exemple de requête :**

```bash
curl -X POST http://15.224.61.131:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0, "TotalCharges": 1020.0
  }'
```

---

## Interface Utilisateur

Interface web moderne (dark theme) accessible depuis un navigateur :

- **Formulaire en 3 étapes** : Profil Client → Services → Contrat & Paiement
- **Champs dynamiques** : options internet/téléphonie adaptées selon les sélections
- **Résultat visuel** : jauge de risque animée + barres de probabilité
- **Historique de session** : suivi des prédictions effectuées
- **Export CSV** : téléchargement de l'historique complet avec toutes les features

---

## Déploiement Local

### Prérequis

- Python 3.11+
- AWS credentials configurées
- Accès au serveur MLflow (`http://15.224.61.131:5000`)

### Installation

```bash
git clone https://github.com/toubaabdellahi/MLOps-projet-final-groupe1.git
cd MLOps-projet-final-groupe1
pip install -r requirements-api.txt
```

### Lancer l'API

```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Interface accessible sur : `http://localhost:8000/ui`

### Relancer le pipeline ML

```bash
dvc pull          # Récupérer les artefacts depuis S3
dvc repro         # Exécuter les 4 stages
dvc push          # Pousser les nouveaux artefacts
```

---

## Déploiement Production (EC2)

```bash
# Démarrer MLflow
nohup mlflow server \
  --host 0.0.0.0 --port 5000 \
  --default-artifact-root s3://mlops-churn-groupe-1/mlflow-artifacts \
  --cors-allowed-origins "*" &

# Construire et lancer le conteneur Docker
docker build -t churn-api .
docker run -d -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://15.224.61.131:5000 \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -e AWS_DEFAULT_REGION=eu-west-3 \
  churn-api
```

---

## CI/CD GitHub Actions

| Workflow | Déclencheur | Actions |
|----------|-------------|---------|
| `deploy.yml` | Merge vers `main` | SSH → EC2, git pull, docker build, docker restart, health check |
| `retrain.yml` | Manuel / cron / modification `.dvc` | dvc repro, MLflow log, docker restart |

Les credentials AWS et la clé SSH sont stockés dans **GitHub Secrets** — aucun secret dans le code.

---

## Structure du Repository

```
├── api/
│   └── main.py              # FastAPI — endpoints /predict, /ui, /reload
├── src/
│   ├── preprocess.py        # Stage DVC 1 — prétraitement
│   ├── train.py             # Stage DVC 2 — entraînement
│   ├── evaluate.py          # Stage DVC 3 — évaluation
│   └── register.py          # Stage DVC 4 — enregistrement MLflow
├── static/
│   └── index.html           # Interface web (dark theme, 3 étapes, export CSV)
├── .github/workflows/
│   ├── deploy.yml           # Workflow CI/CD déploiement
│   └── retrain.yml          # Workflow CI/CD retraining
├── dvc.yaml                 # Définition du pipeline DVC
├── Dockerfile               # Image Docker pour l'API
└── requirements-api.txt     # Dépendances Python
```

---

## Infrastructure AWS

- **EC2** : t3.medium — Ubuntu — IP publique `15.224.61.131`
- **S3** : `s3://mlops-churn-groupe-1/` — données raw, processed, MLflow artifacts, DVC store
- **Ports ouverts** : 22 (SSH), 5000 (MLflow), 8000 (API)
