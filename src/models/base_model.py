import abc
import yaml
from joblib import dump
import pandas as pd
from src.utils import paths

class BaseModel(abc.ABC):
    def __init__(self, df: pd.DataFrame, config=None):
        
        self.df = df

        if config is None:
            with open(str(paths.project_root) + '/config/models_configs.yaml', 'r') as file:
                self.config = yaml.safe_load(file)
        else:
            self.config = config

    @abc.abstractmethod
    def optimize(self, n_trials):
        pass

    @abc.abstractmethod
    def fit(self, x_train, y_train):
        pass

    @abc.abstractmethod
    def predict(self, x_test):
        pass

    @abc.abstractmethod
    def save_model(self, model_name, model):
        save_path = paths.project_root / f"trained_models/{model_name}.pkl"
        dump(model, save_path)

    @abc.abstractmethod
    def load_model(self, path):
        pass