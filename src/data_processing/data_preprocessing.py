import logging
import pandas as pd
from src.data_processing.features_engineering_bib import FeaturesEngineering

def preprocessing(data_file: str):
    logger = logging.getLogger(__name__)
    logger.info("Начало подготовки к обучению")

    df = pd.read_csv(data_file, encoding='utf-8')

    fe = FeaturesEngineering(df)

    steps = [
        fe.label_encoder,
        fe.normalization,
        fe.standartization
    ]

    logger.info("Начало обработки данных")

    for step in steps:
        df = step()
    
    logger.info("Обработка завершена")

    return df