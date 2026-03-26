import joblib
from pathlib import Path
import logging
from typing import Dict, Any

from src.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_model_artifacts: Dict[str, Any] = None

def load_model():
    """
    Загрузка модели из файла
    """
    global _model_artifacts

    if _model_artifacts is not None:
        return _model_artifacts

    model_path = config.model_path

    if not model_path.exists():
        raise FileNotFoundError("Model not found. Please check path or train model first")

    logger.info(f"Loading model from {model_path}")
    _model_artifacts = joblib.load(model_path)

    required_keys = ["model", "preprocessor", "metrics", "config", "feature_names", "metadata"]
    for key in required_keys:
        if key not in _model_artifacts.keys():
            logger.warning(f"Missing required key {key}")

    logger.info(f"Model loaded successfully")

    return _model_artifacts

def get_model():
    """
    Получить модель
    """
    global _model_artifacts

    if _model_artifacts is not None:
        return _model_artifacts.get("model")

    return load_model().get("model")

def get_preprocessor():
    """
    Получить препроцессор
    """
    global _model_artifacts
    if _model_artifacts is not None:
        return _model_artifacts.get("preprocessor")

    return load_model().get("preprocessor")

def get_threshold():
    """
    Получить порог
    """
    global _model_artifacts

    if _model_artifacts is not None:
        if _model_artifacts.get("config", {}).get("threshold", None) is not None:
            return _model_artifacts["config"]["threshold"]
        else:
            raise ValueError("Threshold not save. Please check are you train model or not?")

    return load_model().get('config', {}).get('threshold', 0.4)

def get_artifact():
    """
    Получить артефакт
    """
    global _model_artifacts

    if _model_artifacts is not None:
        return _model_artifacts

    return load_model()
