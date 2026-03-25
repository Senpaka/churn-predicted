
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator, ValidationError, ConfigDict
from typing import List, Optional

class UserFeatures(BaseModel):
    """
    Признаки пользователя
    """
    RowNumber: int = Field(..., ge=0, description="Row number")
    CustomerId: int = Field(..., ge=0, description="Customer ID")
    Surname: str = Field(..., description="Surname")
    CreditScore: float = Field(..., ge=300, le=850)
    Geography: str = Field(..., description="Geography: France, Germany, Spain")
    Gender: str = Field(..., description="Gender: Male, Female")
    Age: int = Field(..., ge=18, le=100)
    Tenure: int = Field(..., ge=0, le=10)
    Balance: float = Field(..., ge=0)
    NumOfProducts: int = Field(..., ge=1, le=4)
    HasCrCard: int = Field(..., ge=0, le=1)
    IsActiveMember: int = Field(..., ge=0, le=1)
    EstimatedSalary: float = Field(..., gt=0)
    SatisfactionScore: int = Field(..., ge=1, le=5, alias="Satisfaction Score")
    PointEarned: int = Field(..., ge=0, le=1000, alias="Point Earned")
    CardType: str = Field(..., alias="Card Type", description="Card Type: GOLD, PLATINUM, SILVER, DIAMOND")

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "RowNumber": 1,
                "CustomerId": 15634602,
                "Surname": "Senpaka",
                "CreditScore": 619,
                "Geography": "France",
                "Gender": "Female",
                "Age": 42,
                "Tenure": 2,
                "Balance": 0,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 101348.88,
                "SatisfactionScore": 2,
                "PointEarned": 464,
                "CardType": "DIAMOND"
            }
        }
    )


    @field_validator("Geography")
    @classmethod
    def validate_geography(cls, v):
        allowed = ["France", "Germany", "Spain"]

        if v not in allowed:
            raise ValueError("Geography must be one of {}".format(allowed))
        return v

    @field_validator("Gender")
    @classmethod
    def validate_surname(cls, v):
        allowed = ["Male", "Female"]

        if v not in allowed:
            raise ValueError("Gender must be one of {}, exists only 2 gender!!!!!".format(allowed))

        return v

    @field_validator("CardType")
    @classmethod
    def validate_age(cls, v):
        allowed = ["GOLD", "PLATINUM", "SILVER", "DIAMOND"]

        if v not in allowed:
            raise ValueError("Card type must be one of {}".format(allowed))

        return v

class BathUserFeatures(BaseModel):
    """
    Батч пользователей
    """
    users: List[UserFeatures]

class PredictionResponse(BaseModel):
    """
    Ответ предсказания
    """
    customer_id: Optional[int] = Field(None, description="The customer id of the user")
    churn_probability: float = Field(..., ge=0, le=1, description="The prediction of the user")
    prediction: str = Field(..., description="The prediction of the user (churn or stay)")
    risk_level: str = Field(..., description="The risk level of the user (high, mid, low)")
    timestamp: str = Field(..., description="The timestamp of prediction")

class BathPredictionResponse(BaseModel):
    """
    Ответ на батч предсказания
    """
    predictions: List[PredictionResponse]
    total: int
    churn_count: int
    timestamp: str

class HealthResponse(BaseModel):
    """
    Ответ на состояние Api
    """
    status: str
    model_loaded: bool
    model_version: str
    timestamp: str



