import os
import pickle
import threading
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://15.224.61.131:5000")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "churn-prediction-model")
MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")

SCALER_PATH = os.getenv("SCALER_PATH", "data/processed/scaler.pkl")
ENCODERS_PATH = os.getenv("ENCODERS_PATH", "data/processed/label_encoders.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "data/processed/feature_names.pkl")

_model = None
_scaler = None
_label_encoders = None
_feature_names = None
_model_lock = threading.Lock()
_load_error = None


def _load_artifacts():
    global _scaler, _label_encoders, _feature_names
    with open(SCALER_PATH, "rb") as f:
        _scaler = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        _label_encoders = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        _feature_names = pickle.load(f)
    print(f"  Preprocessing artifacts loaded — {len(_feature_names)} features")


def _load_model():
    global _model, _load_error
    with _model_lock:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
            _model = mlflow.sklearn.load_model(uri)
            _load_error = None
            print(f"  Model loaded: {type(_model).__name__} from {uri}")
        except Exception as e:
            _load_error = str(e)
            print(f"  Model load failed: {e}")


app = FastAPI(title="Churn Prediction API", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
def startup():
    print("Loading preprocessing artifacts...")
    _load_artifacts()
    print(f"Loading model from MLflow ({MLFLOW_TRACKING_URI})...")
    t = threading.Thread(target=_load_model, daemon=True)
    t.start()


class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability_churn: float
    probability_no_churn: float


@app.get("/", summary="Health check")
def health_check():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "stage": MODEL_STAGE,
        "model_loaded": _model is not None,
        "mlflow_uri": MLFLOW_TRACKING_URI,
    }


@app.get("/ui", response_class=HTMLResponse, summary="Web UI")
def ui():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


@app.post("/reload", summary="Reload model from MLflow Registry")
def reload_model():
    t = threading.Thread(target=_load_model, daemon=True)
    t.start()
    return {"status": "reload started", "model": MODEL_NAME, "stage": MODEL_STAGE}


@app.post("/predict", response_model=PredictionResponse, summary="Predict churn")
def predict(features: CustomerFeatures):
    if _model is None:
        if _load_error:
            raise HTTPException(status_code=503, detail=f"Model unavailable: {_load_error}")
        raise HTTPException(status_code=503, detail="Model loading in progress, retry in a few seconds")

    data = features.model_dump()
    df = pd.DataFrame([data])

    for col, le in _label_encoders.items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col])
            except ValueError:
                df[col] = 0

    df = df[_feature_names]
    X_scaled = _scaler.transform(df)

    prediction = int(_model.predict(X_scaled)[0])
    probabilities = _model.predict_proba(X_scaled)[0]

    return PredictionResponse(
        prediction=prediction,
        prediction_label="Churn" if prediction == 1 else "No Churn",
        probability_churn=float(probabilities[1]),
        probability_no_churn=float(probabilities[0]),
    )
