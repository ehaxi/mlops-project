import mlflow
import mlflow.sklearn
import mlflow.catboost
import mlflow.xgboost

# Указываем URI сервера, запущенного через start_mlflow_server.py
mlflow.set_tracking_uri("http://127.0.0.1:5000")


def start_mlflow_run(experiment_name: str = "Default"):
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run()


def log_params(params: dict):
    mlflow.log_params(params)


def log_metrics(metrics: dict):
    mlflow.log_metrics(metrics)


def log_model(model, model_name: str = "model"):
    loggers = {
        "CatBoostClassifier": mlflow.catboost.log_model,
        "XGBClassifier": mlflow.xgboost.log_model,
        "LogisticRegression": mlflow.sklearn.log_model,
        "RandomForestClassifier": mlflow.sklearn.log_model
    }

    model_class_name = model.__class__.__name__

    if model_class_name not in loggers:
        raise ValueError(f"Данный тип модели не поддерживается: {model_class_name}")

    loggers[model_class_name](model, artifact_path=model_name)


def log_artifact(path_to_file: str):
    mlflow.log_artifact(path_to_file)


def load_model():
    # Заглушка, можешь реализовать позже
    pass