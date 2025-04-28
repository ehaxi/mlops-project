from datetime import datetime
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
from joblib import dump
from src.models.base_model import BaseModel
from src.utils import mlflow_utils, paths

class CatBoostClf(BaseModel):
    def __init__(self, df: pd.DataFrame, splits: list, config=None):
        super().__init__(df, splits, config)
        
        params = self.config['models']['catboost_clf']
        self.splits = splits

        self.model = CatBoostClassifier(**params, train_dir=None)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_name = f"{self.model.__class__.__name__}_{self.timestamp}"

    def fit(self):
        with mlflow_utils.start_mlflow_run(str(f"CatBoostClf_{self.timestamp}")):
            mlflow_utils.log_params(self.config['models']['catboost_clf'])

            for x_train, x_test, y_train, y_test in self.splits:
                self.model.fit(x_train, y_train)
                self.evaluate(x_test, y_test)

            mlflow_utils.log_model(self.model)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def evaluate(self, x_test, y_test):
        predictions = self.predict(x_test)
        
        # Опционально изменить на метрики, которые будут храниться в .yaml
        report = classification_report(y_test, predictions, output_dict=True)

        flat_metrics = {}
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    flat_metrics[f"{label}_{metric_name}"] = value
            else:
                flat_metrics[label] = metrics

        mlflow_utils.log_metrics(flat_metrics)

    def save_model(self):
        output_path_bydump = str(paths.project_root) + f'/trained_models/{self.model_name}.pkl'

        dump(self.model, output_path_bydump)

    def load_model(self):
        pass