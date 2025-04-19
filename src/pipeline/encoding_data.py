import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def label_encoder(df: pd.DataFrame):
    logger = logging.getLogger(__name__)

    logger.info("Начало работы энкодера типа Label Encoding")

    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    corr = df.corr()

    logger.info("Данные закодированы")

    return corr