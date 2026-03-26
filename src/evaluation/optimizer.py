import logging
import numpy as np
from sklearn.metrics import recall_score, precision_score
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier
import logging

from src.preprocess import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.config import config



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

class RegularizationOptimizer:

    def find_best_regularization(self, df: pd.DataFrame, cv_folds=5) -> pd.DataFrame:

        preprocessor = DataPreprocessor()

        df_train, df_test = train_test_split(
            df,
            test_size=0.25,
            random_state=config.random_state,
            stratify=df['Exited']
        )

        X_train, y_train = preprocessor.fit_transform(df_train)
        X_test, y_test = preprocessor.transform(df_test, return_target=True)

        reg_alphas = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
        reg_lambdas = [0.1, 0.2, 0.5, 0.8, 0.9]
        #scale_pos_weight = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 8, 10]

        base_weight = 3.905
        scale_pos_weight = [w * base_weight for w in [0.5, 1, 1.5]]

        max_depth = [3, 4, 5]
        learning_rate = [0.01, 0.05]
        n_estimators = [100, 150]
        subsample = [0.8, 0.9, 1.0]

        model = XGBClassifier(
            random_state=config.random_state,
            eval_metric=config.eval_metric,
            verbosity=config.verbosity,
        )

        grid_params = {
            'reg_alpha': reg_alphas,
            'reg_lambda': reg_lambdas,
            'scale_pos_weight': scale_pos_weight,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'subsample': subsample,
        }

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=grid_params,
            cv=cv_folds,
            scoring="recall",
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best params: {grid_search.best_params_}")
        logger.info(f"Best CV recall: {grid_search.best_score_:.4f}")

        results = pd.DataFrame(grid_search.cv_results_)
        top_results = results.nlargest(5, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]

        for i, row in top_results.iterrows():
            logger.info(f"  {row['params']} -> recall: {row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}")


