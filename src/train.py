
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Optional, Any

from datetime import datetime as dt
import joblib
import logging

from src.config import config
from src.preprocess import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainModel:
    """
    Класс для обучения и сохранения модели предсказания оттока
    """
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.model = None
        self.model_path = None
        self.is_trained = False
        self.training_metrics = None

    def train(
            self,
            df: pd.DataFrame,
            test_size: float = None,
            return_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Обучает модель на данных

        :param df: Исходный DataFrame
        :param test_size: Доля тестовой выборки (если None, берется из config)
        :param return_predictions: Возвращать ли предсказания в результатах
        :return: Словарь с результатами обучения
        """

        logger.info("=" * 60)
        logger.info('Starting model training')
        logger.info("=" * 60)

        if test_size is None and config.test_size is not None:
            test_size = config.test_size
        elif test_size is None and config.test_size is None:
            test_size = 0.2

        df_train, df_test = train_test_split(
            df,
            test_size=test_size,
            random_state=config.random_state,
            stratify=df['Exited']
        )
        logger.info(f"Training data type: {type(df_train)}")
        logger.info(f"Test data type: {type(df_test)}")
        X_train, y_train = self.preprocessor.fit_transform(df_train)
        X_test, y_test = self.preprocessor.transform(df_test, return_target=True)

        logger.info(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

        logger.info("=" * 60)
        logger.info("Training XGBoost model...")
        logger.info("=" * 60)

        self.model = XGBClassifier(
            scale_pos_weight=config.scale_pos_weight,
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            random_state=config.random_state,
            verbosity=config.verbosity,
            eval_metric=config.eval_metric,
            early_stopping_rounds=config.early_stopping_rounds
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= config.threshold).astype(int)
        print(type(y_proba))
        self.training_metrics = {
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "accuracy": accuracy_score(y_test, y_pred),
            "threshold": config.threshold,
            "scale_pos_weight": config.scale_pos_weight,
            "test_size": test_size,
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        self.training_metrics["confusion matrix"] = {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

        logger.info("=" * 60)
        logger.info("TRAINING RESULTS")
        logger.info("=" * 60)
        logger.info(f"precision: {self.training_metrics['precision']}")
        logger.info(f"recall: {self.training_metrics['recall']}")
        logger.info(f"f1: {self.training_metrics['f1']}")
        logger.info(f"accuracy: {self.training_metrics['accuracy']}")
        logger.info(f"confusion matrinx: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        logger.info("=" * 60)

        self.is_trained = True

        results = {
            "model": self.model,
            "preprocessor": self.preprocessor,
            "metrix": self.training_metrics,
            "X_test": X_test,
            "y_test": y_test,
        }

        if return_predictions:
            results["y_pred"] = y_pred
            results["y_proba"] = y_proba

        return results

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предсказывает на новых данных

        :param df: Новый DataFrame
        :return:
            predictions: Массив предсказаний (0/1)
            probabilities: Массив вероятностей
        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train() first")

        X = self.preprocessor.transform(df)

        y_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= config.threshold).astype(int)

        return y_pred, y_proba

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Предсказывает вероятности на новых данных

        :param df: Новый DataFrame
        :return: Массив вероятностей (0/1)
        """
        _, y_proba = self.predict(df)

        return y_proba

    def save(self, path: str = None):
        """
        Сохраняет модель и все артефакты в файл
        :param path: Путь сохранения
        """

        if not self.is_trained:
            raise ValueError("Model is not trained. Train model first")

        path = path or str(config.model_path)

        artifacts = {
            "model": self.model,
            "preprocessor": self.preprocessor,
            "metrix": self.training_metrics,
            "config": {
                "scale_pos_weight": config.scale_pos_weight,
                "n_estimators": config.n_estimators,
                "learning_rate": config.learning_rate,
                "max_depth": config.max_depth,
                "random_state": config.random_state,
                "eval_metric": config.eval_metric,
                "threshold": config.threshold,
                "test_size": config.test_size,
                "verbosity": config.verbosity,
            },
            "feature_names": self.preprocessor.get_features_names(),
            "metadata": {
                "train_data": dt.now().strftime("%d%m%Y-%H%M%S"),
                "version": "0.2.0",
                "description": "Churn prediction model with balanced weights and change threshold"
            }
        }

        joblib.dump(artifacts, path)
        logger.info(f"Model saved to {path}")

    @staticmethod
    def load(path: str) -> Dict[str, Any]:
        """
        Загружает модель и артефакты из файла
        :param path: Путь сохраненной модели
        :return: Словарь с загруженными артефактами
        """
        import os
        if not os.path.exists(path):
            raise FileExistsError("Path does not exist")

        artifacts = joblib.load(path)
        logger.info(f"Model loaded from {path}")

        return artifacts
