import logging
import pandas as pd
from src.data_processing import features_engineering_bib

def preprocessing(data_file: str):
    logger = logging.getLogger(__name__)
    logger.info("Начало подготовки к обучению")

    df = pd.read_csv(data_file, encoding='utf-8')

    steps = [
        features_engineering_bib.label_encoder,
        features_engineering_bib.normalization,
        features_engineering_bib.standartization
    ]

    logger.info("Начало обработки данных")
    
    for step in steps:
        try:
            df = step(df)
        except Exception as exception:
            logger.error(f"Ошибка на этапе {step.__name__}: {exception}")
            raise exception

    try:
        splits = features_engineering_bib.data_separation(df)
        logger.info("Разделение данных завершено")
    except Exception as exception:
        logger.error(f"Ошибка при разделении данных: {exception}")
        raise exception
    
    logger.info("Обработка завершена")

    return df, splits