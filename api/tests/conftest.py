import pytest
from fastapi.testclient import TestClient
import pandas as pd
import tempfile
import json
from pathlib import Path

from api.app import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_customer():
    """Пример валидного клиента"""
    return {
        "RowNumber": 1,
        "CustomerId": 15634602,
        "Surname": "Hargrave",
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

@pytest.fixture
def sample_customer_high_risk():
    """Пример клиента с высоким риском"""
    return {
        "RowNumber": 2,
        "CustomerId": 15647311,
        "Surname": "Hill",
        "CreditScore": 608,
        "Geography": "Spain",
        "Gender": "Female",
        "Age": 55,
        "Tenure": 1,
        "Balance": 83807.86,
        "NumOfProducts": 3,
        "HasCrCard": 0,
        "IsActiveMember": 0,
        "EstimatedSalary": 112542.58,
        "SatisfactionScore": 3,
        "PointEarned": 456,
        "CardType": "DIAMOND"
    }

@pytest.fixture
def sample_customers_batch(sample_customer, sample_customer_high_risk):
    """Пример пакета клиентов"""
    return {
        "users": [sample_customer, sample_customer_high_risk]
    }

@pytest.fixture
def sample_csv_file():
    """Создает временный CSV файл с тестовыми данными"""
    df = pd.DataFrame([
        {
            "RowNumber": 1, "CustomerId": 15634602, "Surname": "Hargrave",
            "CreditScore": 619, "Geography": "France", "Gender": "Female",
            "Age": 42, "Tenure": 2, "Balance": 0, "NumOfProducts": 1,
            "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 101348.88,
            "Satisfaction Score": 2, "Point Earned": 464, "Card Type": "DIAMOND"
        },
        {
            "RowNumber": 2, "CustomerId": 15647311, "Surname": "Hill",
            "CreditScore": 608, "Geography": "Spain", "Gender": "Female",
            "Age": 55, "Tenure": 1, "Balance": 83807.86, "NumOfProducts": 3,
            "HasCrCard": 0, "IsActiveMember": 0, "EstimatedSalary": 112542.58,
            "Satisfaction Score": 3, "Point Earned": 456, "Card Type": "DIAMOND"
        }
    ])

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        yield f.name

    Path(f.name).unlink()