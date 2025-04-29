import mlflow
import mlflow.catboost
from mlflow.tracking import MlflowClient

def start_mlflow_run(run_name: str):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    client = MlflowClient()

    experiment = client.get_experiment_by_name(name="Heart_diseases_clf")
    if experiment and experiment.lifecycle_stage == "deleted":
        client.restore_experiment(experiment.experiment_id)

    mlflow.set_experiment("Heart_diseases_clf")

    return mlflow.start_run(run_name=run_name)


def log_params(params: dict):
    mlflow.log_params(params)


def log_metrics(metrics: dict):
    mlflow.log_metrics(metrics)


def log_model(model):
    loggers = {
        "CatBoostClassifier": mlflow.catboost.log_model,
    }

    model_class_name = model.__class__.__name__

    if model_class_name not in loggers:
        raise ValueError(f"Данный тип модели не поддерживается: {model_class_name}")

    loggers[model_class_name](model, model_class_name)


def log_artifact(file, name: str):
    mlflow.log_artifact(file, name)


def load_model():
    # Заглушка, можешь реализовать позже
    pass