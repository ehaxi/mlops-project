import abc
import pandas as pd
import numpy as np

class BaseModel(abc.ABC):
    def __init__(self, df: pd.DataFrame, path: str, 
                 x_train: np.ndarray, x_test: np.ndarray,
                 y_train: list, y_test: list):
        
        self.df = df
        self.path = path

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    @abc.abstractmethod
    def fit(self, x_train, y_train):
        pass

    @abc.abstractmethod
    def predict(self, x_test):
        pass

    @abc.abstractmethod
    def evaluate(self, x_test, y_test):
        pass

    def save_model(self, path):
        pass

    def load_model(self, path):
        pass