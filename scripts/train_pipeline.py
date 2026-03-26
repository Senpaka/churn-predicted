
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import logging
from src.config import config
from src.train import TrainModel
from src.explain.shap_explainer import ModelExplainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Полный пайплайн модели

    :return: Артефакты модели
    """
    logger.info("=" * 60)
    logger.info("Churn prediction pipeline")
    logger.info("=" * 60)
    logger.info("Loading data")
    if not config.data_path.exists():
        logger.error("Path does not exist")
        logger.info("Place data.csv in data/ directory")
        return

    df = pd.read_csv(config.data_path)

    logger.info(f"Data loaded: {df.shape[0]} rows and {df.shape[1]} columns")

    trainer = TrainModel()
    results = trainer.train(df, return_predictions=True)

    trainer.save()

    logger.info("=" * 60)
    logger.info("Pipeline completed")
    logger.info("=" * 60)
    logger.info(f"Model saved to {config.model_path}")
    logger.info("=" * 60)
    logger.info("Final metrics:")
    logger.info("=" * 60)
    logger.info(f"recall: {results['metrics']['recall']:.4f}")
    logger.info(f"precision: {results['metrics']['precision']:.4f}")
    logger.info(f"f1-score: {results['metrics']['f1']:.4f}")
    logger.info(f"accuracy: {results['metrics']['accuracy']:.4f}")
    logger.info("=" * 60)

    return results

if __name__ == "__main__":

    results = main()

    logger.info("Cross validation metrics:")
    logger.info("=" * 30)
    logger.info(f"{'Metric':<12} | {'Mean':>8} | {'Std':>8}")
    logger.info("-" * 30)
    cv_metrics = results["cv_metrics"]
    for metric, stats in cv_metrics.items():
        mean = stats["mean"]
        std = stats["std"]
        logger.info(f"{metric:<12} | {mean:>8.4f} | {std:>8.4f}")
    logger.info("=" * 30)

    shap_df = ModelExplainer.explain(results["model"], results["X_test"])

    logger.info(f"\n{shap_df.abs().mean().sort_values(ascending=False).head(10)}")




