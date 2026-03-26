import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional, Any
import logging

from src.config import config
from src.features import FeaturesEngineering

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Класс для полной предобработки данных

    Включает:
        -feature engineering
        -Масштабирование числовых признаков
        -Разделение на X и y
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.engineering = FeaturesEngineering()
        self.features_names_ = None
        self.numerical_features = config.numerical_features
        self.columns_to_drop = config.pre_train_drop_features
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame, target_col: str = "Exited") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Обучает препроцессор и преобразует данные

        :param df: Исходный DataFrame
        :param target_col: название целевого переменной
        :return:
            X: DataFrame с признаками
            y: Series с целевой переменной
        """
        logger.info("=" * 60)
        logger.info(f"Fitting preprocessor")
        logger.info("=" * 60)

        df_processed = self.engineering.create_features(df)

        y = df_processed[target_col]
        X = df_processed.drop([target_col] + self.columns_to_drop, axis=1).copy()

        logger.info(f"Total feature after preprocessing: {len(X.columns)}")

        self.features_names_ = X.columns.tolist()

        features_to_scaler = [col for col in self.numerical_features if col in X.columns]

        if features_to_scaler:
            for col in features_to_scaler:
                X[col] = X[col].astype(float)

            scaled_values = self.scaler.fit_transform(X[features_to_scaler])

            for i, col in enumerate(features_to_scaler):
                X[col] = scaled_values[:, i]

            logger.info("Scaled numerical features")
        else:
            logger.warning("No numerical features found to scaler")

        self.is_fitted = True

        logger.info(f"X shape: {X.shape}")
        logger.info(f"y shape: {y.shape}")
        logger.info(f"Class distribution:\n{y.value_counts(normalize=True)}")

        return X, y

    def transform(self, df: pd.DataFrame, target_col: str = "Exited", return_target: bool = False) -> pd.DataFrame | Tuple[pd.DataFrame, pd.Series]:
        """
        Преобразует данные (без переобучения)

        :param df: Исходный DataFrame
        :param target_col: Название целевой переменной
        :param return_target: True если требуется вернуть целевую переменную, иначе False
        :return:
            X: DataFrame с признаками, готовыми для предсказания
            y: (опционально) Series с целевыми переменными
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor is not fitted. Call fit_transform() first")

        logger.info("=" * 60)
        logger.info(f"Transforming data")
        logger.info("=" * 60)

        if return_target:
            y = df[target_col]

        df_processed = self.engineering.create_features(df)

        X = df_processed[self.features_names_].copy()

        features_to_scaler = [col for col in self.numerical_features if col in X.columns]
        if features_to_scaler:
            for col in features_to_scaler:
                X[col] = X[col].astype(float)

            scaled_values = self.scaler.transform(X[features_to_scaler])

            for i, col in enumerate(features_to_scaler):
                X[col] = scaled_values[:, i]
        else:
            logger.warning("No numerical features found to scaler")

        logger.info(f"Transform data shape: {X.shape}")

        if return_target:
            return X, y

        return X

    def get_features_names(self) -> List[str]:
        """
        Возвращает названия признаков

        :return: Список названий признаков
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor is not fitted. Call fit_transform() first")
        return self.features_names_

    def save(self, path: str):
        """
        Сохраняет препроцессор в файл

        :param path: Путь для сохранения
        """
        import joblib

        artifacts = {
            "features": self.get_features_names(),
            "feature_engineering": self.engineering,
            "scaler": self.scaler,
            "numerical_features": self.numerical_features,
            "_is_fitted": self.is_fitted
        }
        joblib.dump(artifacts, path)
        logger.info(f"Saved preprocessor to {path}")

    @staticmethod
    def load(path: str) -> Any:
        """
        Загружает препроцессор из файла

        :param path: Путь до препроцессора
        :return: Готовый препроцессор
        """
        import joblib

        artifacts = joblib.load(path)

        preprocessor = DataPreprocessor()
        preprocessor.features_names_ = artifacts["features"]
        preprocessor.engineering = artifacts["feature_engineering"]
        preprocessor.scaler = artifacts["scaler"]
        preprocessor.numerical_features = artifacts["numerical_features"]
        preprocessor.is_fitted = artifacts["_is_fitted"]

        logger.info(f"Load preprocessor from {path}")

        return preprocessor


