from fastapi import FastAPI, APIRouter
from starlette.middleware.cors import CORSMiddleware

import api.routers.health as health
import api.routers.predict as predict

from api.schemas import (
    UserFeatures,
    BathUserFeatures,
    PredictionResponse,
    BathPredictionResponse,
    HealthResponse
)

app = FastAPI(
    title="Churn Prediction API",
    version="1.0",
    description="API for demonstrate churn prediction"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Total-Rows", "X-Churns-Count"],
)

app.include_router(health.router)
app.include_router(predict.router)


@app.get("/", tags=["Root"])
def root():
    return {
        "Service": "Churn Prediction API",
        "version": "1.0",
        "description": "API for demonstrate churn prediction",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health"
    }


@app.get("/info", tags=["Root"])
def info():
    """Информация о модели"""
    try:
        from api.dependencies import get_artifact
        artifacts = get_artifact()

        return {
            "model_version": artifacts.get("metadata", {}).get("version", "unknown"),
            "threshold": artifacts.get("config", {}).get("threshold", 0.4),
            "features_count": len(artifacts.get("feature_names", [])),
            "metrics": artifacts.get("metrics", {}),
            "trained_date": artifacts.get("metadata", {}).get("train_data", "unknown")
        }
    except Exception as e:
        return {
            "model_version": "not_loaded",
            "error": str(e)
        }