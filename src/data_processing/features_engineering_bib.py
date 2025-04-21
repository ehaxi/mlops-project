import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

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


def data_separation(df: pd.DataFrame):
    # Нужно вручную настроить features и target 
    # Признаки RestingBP, RestingECG удаляются из-за слишком маленькой корреляции с таргетом
    logger.info("Начало разделения данных на тренировочную и тестовую выборки")

    features = df[df.columns.drop(['HeartDisease','RestingBP','RestingECG'])].to_numpy()
    target = df['HeartDisease'].to_list()
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.20, random_state = 2)

    logger.info("Данные разделены")

    return x_train, x_test, y_train, y_test