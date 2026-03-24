from io import BytesIO

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
import pandas as pd
from typing import Any, Dict
from datetime import datetime

from starlette.responses import StreamingResponse

from api.schemas import UserFeatures, PredictionResponse, BathPredictionResponse, BathUserFeatures
from api.dependencies import get_artifact
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["prediction"],
)

def make_prediction(features: UserFeatures, model_artifacts: Dict) -> PredictionResponse:
    """
    Выполнение предсказания

    :param features: Признаки
    :param model_artifacts: Сохраненные артефакты модели
    :return: ответ предсказания
    """
    customer_id = features.CustomerId
    customer_dict = features.model_dump(by_alias=True)

    df_raw = pd.DataFrame([customer_dict])

    logger.info(f"Raw data shape: {df_raw.shape}")
    logger.info(f"Raw data columns: {list(df_raw.columns)}")

    if model_artifacts is None:
        raise HTTPException(status_code=400, detail="Model artifacts not provided. Please load or train model")

    preprocessor = model_artifacts.get("preprocessor")
    model = model_artifacts.get("model")
    threshold = model_artifacts.get("config").get("threshold")

    X = preprocessor.transform(df_raw)

    probability = model.predict_proba(X)[0,1]
    print(probability)

    prediction = (probability >= threshold).astype(int)

    if probability > 0.6:
        risk = "high risk"
    elif probability >= threshold:
        risk = "mid risk"
    else:
        risk = "low risk"

    return PredictionResponse(
        customer_id=customer_id,
        churn_probability=round(probability, 4),
        prediction="churn" if prediction == 1 else "stay",
        risk_level=risk,
        timestamp=datetime.now().isoformat()
    )

@router.post("/prediction", response_model=PredictionResponse)
def predict(
        user: UserFeatures,
        model_artifacts: Dict[str, Any] = Depends(get_artifact),
):
    """
    Делает предсказание для 1 пользователя
    """
    return make_prediction(user, model_artifacts)

@router.post("/prediction_batch", response_model=BathPredictionResponse)
def predict_batch(
        batch: BathUserFeatures,
        model_artifacts: Dict[str, Any] = Depends(get_artifact),
):
    """
    Делает предсказание для многих пользователей
    """

    count_curn = 0
    results = []

    for user in batch.users:
        result = make_prediction(user, model_artifacts)
        results.append(result)

        if result.prediction == "churn":
            count_curn += 1

    return BathPredictionResponse(
        predictions=results,
        total=len(results),
        churn_count=count_curn,
        timestamp=datetime.now().isoformat()
    )

@router.post("/prediction_from_csv")
async def predict_from_csv(
        file: UploadFile = File(..., description="CSV file with user data"),
        model_artifacts: Dict[str, Any] = Depends(get_artifact)
):
    """
    Делает предсказание под данным из .csv файла
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="File extension not supported",
        )

    try:
        content = await file.read()
        df = pd.read_csv(BytesIO(content))
        logger.info(f"Processing file {file.filename}, rows {len(df)}, columns {df.columns}")

        model = model_artifacts.get("model")
        threshold = model_artifacts.get("config", {}).get("threshold", 0.4)
        preprocessor = model_artifacts.get("preprocessor")
        features_names = model_artifacts.get("feature_names")

        X = preprocessor.transform(df)

        if features_names:
            missing = set(features_names) - set(X.columns)
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail="Missing features",
                )

        probability = model.predict_proba(X)[:, 1]
        prediction = (probability >= threshold).astype(int)

        risk_level = np.where(
            probability >= 0.6, "high risk", np.where(
                probability >= threshold, "mid risk", "low risk"
            )
        )

        result_df = df.copy()

        result_df["probability"] = probability.round(4)
        result_df["prediction"] = np.where(prediction == 1, "churn", "stay")
        result_df["risk_level"] = risk_level
        result_df["timestamp"] = datetime.now().isoformat()

        total = len(result_df)
        churn_count = prediction.sum()

        logger.info(f"Prediction completed: total={total}, churn_count={churn_count}, churn_rate={churn_count/total:.2%}")

        output = BytesIO()
        result_df.to_csv(output, index=False, encoding="utf-8-sig")
        output.seek(0)

        timestamp = datetime.now().isoformat()
        output_name = f"prediction_{timestamp}.csv"

        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={output_name}",
                "X-Total-rows": str(total),
                "X-Churns-Count": str(churn_count)
            }
        )

    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail="File is empty",
        )
    except pd.errors.ParserError:
        raise HTTPException(
            status_code=400,
            detail="Invalid file format"
        )
    except Exception as e:
        logger.error(f"File prediction error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {e}"
        )




