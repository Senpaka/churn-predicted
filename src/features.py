
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import logging

from src.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeaturesEngineering:

    def __init__(self):
        self.numerical_features = config.numerical_features
        self.categorical_features = config.categorical_features
        self.columns_to_drop = config.columns_to_drop
        self.age_bins = config.age_bins
        self.age_labels = config.age_labels
        self.feature_names_ = None
        self._target = None

    def create_features(self, df: pd.DataFrame, target: str = "Exited") -> pd.DataFrame:
        self._target = target

        logger.info("=" * 60)
        logger.info(f"Start creating features")
        logger.info("=" * 60)

        df = df.copy()

        cols_to_drop = [col for col in self.columns_to_drop if col in df.columns]
        df = df.drop(cols_to_drop, axis=1)
        logger.info(f"Dropping columns: {cols_to_drop}")

        df["Age_Group"] = pd.cut(
            df["Age"],
            bins=self.age_bins,
            labels=self.age_labels,
            right=False,
            include_lowest=True
        )
        logger.info(f"Created feature: 'Age_Group'")

        categorial_cols_to_drop = [col for col in self.categorical_features if col in df.columns]
        if categorial_cols_to_drop:
            df = pd.get_dummies(
                df,
                columns=categorial_cols_to_drop,
                drop_first=True,
                dtype=int
            )
            logger.info(f"one-hot encoded columns: {categorial_cols_to_drop}")

        df["Has_Balance"] = (df["Balance"] == 0).astype(int)
        logger.info(f"Created feature: 'Has_Balance'")

        exclude_for_names = [self._target, "Age", "Exited", "Complain"]
        if "Geography_Readable" in df.columns:
            df = df.drop("Geography_Readable", axis=1)

        self.feature_names_ = [col for col in df.columns if col not in exclude_for_names]
        logger.info(f"Feature engineering complete. Total features: {len(self.feature_names_)}")

        return df

    def get_features_names(self, df: Optional[pd.DataFrame] = None, target: str = "Exited") -> List[str]:

        if self.feature_names_ is not None:
            return self.feature_names_

        if df is not None:
            exclude_for_names = [target, "Age", "Exited", "Complain"]
            return [col for col in df.columns if col not in exclude_for_names]

        raise ValueError("Features name not set. Call create_features() or pass df")

    def validate_features(self, df: pd.DataFrame) -> bool:

        if self.feature_names_ is None:
            raise ValueError("Call create_features() first")

        missing = set(self.feature_names_) - set(df.columns)
        if missing:
            logger.warning(f"Missing features: {missing}")
            return False

        return True