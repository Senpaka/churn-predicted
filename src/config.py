# src/config.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).parent.parent

@dataclass
class Config:

    data_path: Path = ROOT_DIR / "data" / "data.csv"
    model_path: Path = ROOT_DIR / "models" / "churn_model_final.pkl"
    log_path: Path = ROOT_DIR / "log"

    scale_pos_weight: float = 3.9051667756703727
    threshold: float = 0.4
    n_estimators: int = 300
    early_stopping_rounds: int = 20
    max_depth: int = 6
    learning_rate: float = 0.1
    random_state: int = 42
    test_size: float = 0.25
    verbosity: int = 0
    eval_metric: str = 'logloss'

    numerical_features: List[str] = field(default_factory=lambda: [
        'CreditScore', 'Balance', 'EstimatedSalary',
        'Tenure', 'NumOfProducts', 'Satisfaction Score', 'Point Earned'
    ])
    categorical_features: List[str] = field(default_factory=lambda: [
        'Gender', 'Card Type', 'Geography', 'Age_Group'
    ])
    columns_to_drop: List[str] = field(default_factory=lambda: [
        'RowNumber', 'CustomerId', 'Surname'
    ])

    age_bins: List[str] = field(default_factory=lambda: [18, 30, 50, 65, 100])
    age_labels: List[str] = field(default_factory=lambda: ["18-29", "30-49", "50-64", "65+"])

    pre_train_drop_features: List[str] = field(default_factory=lambda: ["Complain", "Age"])

config = Config()