import logging
import numpy as np
from sklearn.metrics import recall_score, precision_score
import pandas as pd



class ThresholdOptimizer:
    """
    Модель для подбора threshold (порога)
    """
    def find_best_threshold(self, y_true: np.ndarray, y_proba: np.ndarray) -> pd.DataFrame:
        """
        Перебирает пороги

        :param y_true: Истинные значения
        :param y_proba: Предсказанные значения
        :return: DataFrame с значениями threshold, recall, precision
        """
        thresholds = np.linspace(0.2, 0.9, 50)

        retult = []

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            recall = recall_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)

            retult.append((threshold, recall, precision))

        return pd.DataFrame(retult, columns=["threshold", "recall", "precision"])