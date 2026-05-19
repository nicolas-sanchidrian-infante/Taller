from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict
import joblib
import pandas as pd
from heart_model_wrapper import HeartDiseaseWrapper
import time
from datetime import datetime

# NUEVO: Importar Prometheus
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

# Inicializar FastAPI
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API para predecir enfermedad cardíaca usando Machine Learning",
    version="1.0.0"
)

# Servir UI estática en /ui (archivo: static/index.html)
app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")

# Variables globales
model = None
wrapper = None
model_loaded_time = None

# NUEVO: Métricas de Prometheus
metrics_registry = CollectorRegistry()

PREDICTIONS_TOTAL = Counter(
    'predictions_total', 
    'Número total de predicciones realizadas',
    ['endpoint'],
    registry=metrics_registry
)

PREDICTION_DURATION = Histogram(
    'prediction_duration_seconds',
    'Duración de las predicciones en segundos',
    ['endpoint'],
    registry=metrics_registry
)

ACTIVE_PREDICTIONS = Gauge(
    'active_predictions',
    'Número de predicciones activas en este momento'
    , registry=metrics_registry
)

PREDICTION_ERRORS = Counter(
    'prediction_errors_total',
    'Número total de errores en predicciones',
    registry=metrics_registry
)

# Cargar modelo al iniciar
@app.on_event("startup")
async def load_model():
    global model, wrapper, model_loaded_time
    try:
        print("📥 Cargando modelo...")
        model = joblib.load("heart_disease_model.joblib")
        wrapper = HeartDiseaseWrapper(model)
        model_loaded_time = datetime.now()
        print("✅ Modelo cargado correctamente")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        raise

# Schemas
class PatientData(BaseModel):
    age: int = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1)
    chest_pain: int = Field(..., ge=1, le=4)
    rest_blood_pressure: int = Field(..., ge=50, le=250)
    cholesterol: int = Field(..., ge=100, le=600)
    fasting_blood_sugar: int = Field(..., ge=0, le=1)
    rest_ecg: int = Field(..., ge=0, le=2)
    max_heart_rate: int = Field(..., ge=60, le=220)
    exercise_angina: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=10)
    slope: int = Field(..., ge=1, le=3)
    vessels: int = Field(..., ge=0, le=3)
    thal: int = Field(..., ge=3, le=7)

class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    confidence: float
    probabilities: Dict[str, float]
    risk_level: str
    inference_time_ms: float

class BatchPredictionRequest(BaseModel):
    patients: List[PatientData]

# Endpoints
@app.get("/")
async def root():
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "info": "/info",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "metrics": "/metrics"
        }
    }

@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_loaded_at": model_loaded_time.isoformat() if model_loaded_time else None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": str(type(model)),
        "model_steps": [step[0] for step in model.steps],
        "features": wrapper.feature_names,
        "n_features": len(wrapper.feature_names),
        "loaded_at": model_loaded_time.isoformat() if model_loaded_time else None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientData):
    if wrapper is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # NUEVO: Incrementar métricas
    ACTIVE_PREDICTIONS.inc()

    try:
        # Medir tiempo
        start_time = time.time()

        patient_dict = patient.dict()
        result = wrapper.predict(patient_dict)

        inference_time = time.time() - start_time
        result['inference_time_ms'] = round(inference_time * 1000, 2)

        # NUEVO: Registrar métricas
        PREDICTIONS_TOTAL.labels(endpoint='predict').inc()
        PREDICTION_DURATION.labels(endpoint='predict').observe(inference_time)

        return result

    except Exception as e:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    finally:
        ACTIVE_PREDICTIONS.dec()

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    if wrapper is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    ACTIVE_PREDICTIONS.inc()

    try:
        start_time = time.time()

        patients_list = [patient.dict() for patient in request.patients]
        results = wrapper.predict_batch(patients_list)

        inference_time = time.time() - start_time

        # NUEVO: Registrar métricas
        PREDICTIONS_TOTAL.labels(endpoint='predict_batch').inc(len(results))
        PREDICTION_DURATION.labels(endpoint='predict_batch').observe(inference_time)

        return {
            "predictions": results,
            "count": len(results),
            "total_inference_time_ms": round(inference_time * 1000, 2),
            "avg_inference_time_ms": round(inference_time / len(results) * 1000, 2)
        }

    except Exception as e:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")
    finally:
        ACTIVE_PREDICTIONS.dec()

# NUEVO: Endpoint de métricas para Prometheus
@app.get("/metrics")
async def metrics():
    """Expone métricas en formato Prometheus."""
    return Response(content=generate_latest(metrics_registry), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)