import abc
import yaml
import pandas as pd
import numpy as np

class BaseModel(abc.ABC):
    def __init__(self, df: pd.DataFrame, path: str, splits: list, config=None):
        
        self.df = df
        self.path = path

        self.splits = splits

        if config is None:
            with open('.config/models_configs.yaml', 'r') as file:
                self.config = yaml.safe_load(file)
        else:
            self.config = config

    @abc.abstractmethod
    def cross_validation(self, splits):
        pass

    @abc.abstractmethod
    def fit(self, x_train, y_train):
        pass

    @abc.abstractmethod
    def predict(self, x_test):
        pass

    @abc.abstractmethod
    def evaluate(self, x_test, y_test):
        pass

    @abc.abstractmethod
    def save_model(self, path):
        pass

    @abc.abstractmethod
    def load_model(self, path):
        pass