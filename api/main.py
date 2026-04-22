import os
import pickle
import threading
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from typing import Literal
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
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ]
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


@app.get("/model/info", summary="Current model info from MLflow Registry")
def model_info():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    try:
        client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if not versions:
            raise HTTPException(status_code=404, detail="No model in Production stage")
        v = versions[0]
        run = client.get_run(v.run_id)
        return {
            "model_name": MODEL_NAME,
            "version": v.version,
            "stage": MODEL_STAGE,
            "run_id": v.run_id,
            "metrics": run.data.metrics,
            "params": run.data.params,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
