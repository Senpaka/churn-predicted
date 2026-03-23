
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import logging
from src.config import config
from src.train import TrainModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
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
    results = trainer.train(df)

    trainer.save()

    logger.info("=" * 60)
    logger.info("Pipeline completed")
    logger.info("=" * 60)
    logger.info(f"Model saved to {config.model_path}")
    logger.info("=" * 60)
    logger.info("Final metrix:")
    logger.info("=" * 60)
    logger.info(f"recall: {results['metrix']['recall']:.4f}")
    logger.info(f"precision: {results['metrix']['precision']:.4f}")
    logger.info(f"f1-score: {results['metrix']['f1']:.4f}")
    logger.info(f"accuracy: {results['metrix']['accuracy']:.4f}")
    logger.info("=" * 60)

    return results

if __name__ == "__main__":
    main()
