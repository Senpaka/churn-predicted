import shap
from typing import Any
import pandas as pd

class ModelExplainer:
    """
    Модуль для подсчета shap значений
    """
    @staticmethod
    def explain(model: Any, X: pd.DataFrame) -> pd.DataFrame:
        """
        Считает shap (влияние) значения

        :param model: Модель
        :param X: DataFrame с признаками
        :return: DataFrame с названием признаков и shap значением
        """
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_df = pd.DataFrame(
            shap_values,
            columns=X.columns
        )

        return shap_df
