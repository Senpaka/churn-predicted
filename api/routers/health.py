from fastapi import APIRouter, Depends
from api.schemas import HealthResponse
from api.dependencies import get_artifact
from datetime import datetime


router = APIRouter(
    tags=["health"],
    prefix="/health"
)

@router.get("", response_model=HealthResponse)
def health_check(artifact=Depends(get_artifact)):
    """
    Проверка работоспособности API и модели
    """
    model = artifact.get("model", None)
    version = artifact.get("metadata", {}).get("version", 0.0)

    print(artifact.get("feature_names"))

    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=version,
        timestamp=datetime.now().isoformat()
    )