import logging
import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder, 
    MinMaxScaler, 
    StandardScaler
)

class FeaturesEngineering():
    def __init__(self, df):
        self.logger = logging.getLogger(__name__)
        self.df = df

    def label_encoder(self):
        self.logger.info("Начало работы энкодера типа Label Encoding")

        for col in self.df.select_dtypes(include='object').columns:
            self.df[col] = LabelEncoder().fit_transform(self.df[col])

        self.logger.info("Данные закодированы")

        return self.df


    def normalization(self):
        # Нужно вручную выбрать признаки, которые будут подвержены нормализации
        # Нормализация: Oldpeak
        self.logger.info("Начало нормализации")

        scaler = MinMaxScaler()

        self.df['Oldpeak'] = scaler.fit_transform(self.df[['Oldpeak']])

        self.logger.info("Данные нормализованы")

        return self.df


    def standartization(self):
        # Нужно вручную выбрать признаки, которые будут подвержены стандартизации
        # Стандартизация: Age, Cholesterol and MaxHR (RestingBP будет удален)
        self.logger.info("Начало стандартизации")

        scaler = StandardScaler()

        self.df['Age'] = scaler.fit_transform(self.df[['Age']])
        self.df['Cholesterol'] = scaler.fit_transform(self.df[['Cholesterol']])
        self.df['MaxHR'] = scaler.fit_transform(self.df[['MaxHR']])

        self.logger.info("Данные стандартизованны")

        return self.df