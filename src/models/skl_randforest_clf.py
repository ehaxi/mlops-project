from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from base_model import BaseModel
from pipeline import mlflow_utils, paths

class RandForestClf(BaseModel):
    def __init__(self, df: pd.DataFrame, path: str, splits: list, config=None):
        super().__init__(df, path, splits, config)
        
        params = self.config['models']['skl_randforest_clf']

        self.model = RandomForestClassifier(**params)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_name = f"{self.model.__class__.__name__}_{timestamp}"

    def cross_validation(self, splits):
        for x_train, x_test, y_train, y_test in splits:
            self.fit(x_train, y_train)
            self.evaluate(x_test, y_test)

    def fit(self, x_train, y_train):
        with mlflow_utils.start_mlflow_run("sklRandForest Experiment"):
            mlflow_utils.log_params(self.config['models']['skl_randforest_clf'])
            self.model.fit(x_train, y_train)
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

        mlflow_utils.log_metrics(flat_metrics, model_name=self.model_name)

    def save_model(self):
        output_path_bybib = str(paths.project_root) + f'/trained_models/by_modelbib/{self.model_name}.cbm'
        self.model.save_model(output_path_bybib)

        # output_path_bymlflow = str(paths.project_root) + f'/trained_models/by_modelbib/{self.model_name}.cbm'
        # mlflow_utils.log_artifact(output_path_bymlflow)

    def load_model(self):
        pass