import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

def label_encoder(df: pd.DataFrame):
    logger.info("Начало работы энкодера типа Label Encoding")

    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    logger.info("Данные закодированы")

    return df


def normalization(df: pd.DataFrame):
    # Нужно вручную выбрать признаки, которые будут подвержены нормализации
    # Нормализация: Oldpeak
    logger.info("Начало нормализации")

    scaler = MinMaxScaler()

    df['Oldpeak'] = scaler.fit_transform(df['Oldpeak'])

    logger.info("Данные нормализованы")

    return df


def standartization(df: pd.DataFrame):
    # Нужно вручную выбрать признаки, которые будут подвержены стандартизации
    # Стандартизация: Age, Cholesterol and MaxHR (RestingBP будет удален)
    logger.info("Начало стандартизации")

    scaler = StandardScaler()

    df['Age'] = scaler.fit_transform(df['Age'])
    df['Cholesterol'] = scaler.fit_transform(df['Cholesterol'])
    df['MaxHR'] = scaler.fit_transform(df['MaxHR'])

    logger.info("Данные стандартизованны")

    return df


def data_separation(df: pd.DataFrame, n_splits: int = 5):
    # Нужно вручную настроить features и target 
    # Признаки RestingBP, RestingECG удаляются из-за слишком маленькой корреляции с таргетом
    logger.info("Начало разделения данных на тренировочную и тестовую выборки")

    features = df[df.columns.drop(['HeartDisease','RestingBP','RestingECG'])]
    target = df['HeartDisease']

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = []

    for train_index, test_index in skf.split(features, target):
        X_train, X_test = features.iloc[train_index].to_numpy(), features.iloc[test_index].to_numpy()
        y_train, y_test = target.iloc[train_index].to_list(), target.iloc[test_index].to_list()
        splits.append((X_train, X_test, y_train, y_test))

    logger.info("Данные разделены")

    return splits