from datetime import datetime
import time
import logging
import numpy as np
import shutil
import optuna
from optuna.samplers import NSGAIISampler
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    recall_score,
    f1_score,
    precision_recall_curve,
    auc
)
from src.models.base_model import BaseModel
from src.utils import mlflow_utils
from src.data_processing import visualization
from src.utils.paths import project_root


class CatBoostClf(BaseModel):
    def __init__(self, df, config=None):
        super().__init__(df, config)

        self.logger = logging.getLogger(__name__)

        self.model = None
        self.best_params = None
        self.best_trial = None

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_name = f"CatBoostClf_{self.timestamp}"
        self.weights = [0.7, 0.2, 0.1]

    def optimize(self, n_trials=300):
        def log_trial(study, trial):
            """Callback для логирования прогресса оптимизации"""
            trial_time = time.time() - trial.datetime_start.timestamp()
            
            log_message = [
                f"Trial {trial.number}/{n_trials}",
                f"Time: {trial_time:.2f}s",
                f"Recall: {trial.values[0]:.4f}",
                f"F1: {trial.values[1]:.4f}",
                f"PR-AUC: {trial.values[2]:.4f}",
                f"Weighted: {sum(w*v for w,v in zip(self.weights, trial.values)):.4f}"
            ]
            
            if study.best_trials:
                best_score = sum(w*v for w,v in zip(self.weights, study.best_trials[0].values))
                current_score = sum(w*v for w,v in zip(self.weights, trial.values))
                improvement = current_score - best_score
                log_message.append(f"Improvement: {improvement:+.4f}")
            
            self.logger.info(" | ".join(log_message))
            self.logger.debug(f"Params: {trial.params}")

        def objective(trial):
            start_time = time.time()

            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
                'random_strength': trial.suggest_float('random_strength', 1e-9, 10.0, log=True),
                'verbose': False,
                'random_state': 42,
                'auto_class_weights': 'Balanced',
            }

            skf = StratifiedKFold(shuffle=True, random_state=42)

            recall_scores, f1_scores, pr_auc_scores = [], [], []

            features = self.df[self.df.columns.drop(['HeartDisease','RestingBP','RestingECG'])]
            target = self.df['HeartDisease']

            for fold, (train_index, test_index) in enumerate(skf.split(features, target), 1):
                x_train, x_test = features.iloc[train_index].to_numpy(), features.iloc[test_index].to_numpy()
                y_train, y_test = target.iloc[train_index].to_list(), target.iloc[test_index].to_list()
                
                model = CatBoostClassifier(
                    **params,
                    train_dir=None,
                    allow_writing_files=False,
                    early_stopping_rounds=20
                )
                model.fit(
                    x_train, y_train,
                    eval_set=(x_test, y_test),
                    early_stopping_rounds=50,
                    verbose=False
                )

                y_pred = model.predict(x_test)
                y_proba = model.predict_proba(x_test)

                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                precision, recall_pr, _ = precision_recall_curve(y_test, y_proba[:, 1])
                pr_auc = auc(recall_pr, precision)
                
                recall_scores.append(recall)
                f1_scores.append(f1)
                pr_auc_scores.append(pr_auc)

                self.logger.debug(
                    f"Trial {trial.number} | Fold {fold} | "
                    f"Recall: {recall:.4f} | F1: {f1:.4f} | PR-AUC: {pr_auc:.4f}"
                )

            trial.set_user_attr("execution_time", time.time() - start_time)

            return np.mean(recall_scores), np.mean(f1_scores), np.mean(pr_auc_scores)
        
        self.logger.info(f"Starting optimization with {n_trials} trials")
        self.logger.info(f"Metric weights: Recall={self.weights[0]}, F1={self.weights[1]}, PR-AUC={self.weights[2]}")

        study = optuna.create_study(
            directions=["maximize", "maximize", "maximize"],
            sampler=NSGAIISampler(population_size=50)
        )
        study.optimize(objective, n_trials=n_trials, n_jobs=1, callbacks=[log_trial])

        self.best_trial = max(
            study.best_trials,
            key=lambda t: sum(w*v for w,v in zip(self.weights, t.values))
        )

        self.best_params = self.best_trial.params
        self.model = CatBoostClassifier(**self.best_params, train_dir=None)

        exec_time = self.best_trial.user_attrs.get("execution_time", "N/A")
        formatted_params = {
            k: round(v, 5) if isinstance(v, float) else v 
            for k, v in self.best_params.items()
        }
        self.logger.info(
            "Оптимизация завершена\n"
            f"Best trial: #{self.best_trial.number}\n"
            f"Execution time: {exec_time:.2f}s\n"
            f"Recall: {self.best_trial.values[0]:.4f}\n"
            f"F1: {self.best_trial.values[1]:.4f}\n"
            f"PR-AUC: {self.best_trial.values[2]:.4f}\n"
            f"Weighted score: {sum(w*v for w,v in zip(self.weights, self.best_trial.values)):.4f}\n"
            f"Parameters: {formatted_params}"
        )

        self.logger.info("Создание и запись фронта Парето")

        temp_dir = project_root / "temp_results"
        temp_dir.mkdir(exist_ok=True)

        fig_path = temp_dir / f"pareto_front_{self.timestamp}.html"
        fig_pareto_front = visualization.plot_pareto_front(study)
        fig_pareto_front.write_html(fig_path)
        mlflow_utils.log_artifact(str(fig_path), "pareto_front.html")

        fig_path.unlink()
        shutil.rmtree(temp_dir)

    def fit(self):
        self.logger.info("Начало обучения модели")
        start_time = time.time()

        with mlflow_utils.start_mlflow_run(str(f"CatBoostClf_{self.timestamp}")):
            self.logger.info("Запущена оптимизация гиперпараметров модели")
            self.optimize()

            features = self.df[self.df.columns.drop(['HeartDisease','RestingBP','RestingECG'])]
            target = self.df['HeartDisease']

            skf = StratifiedKFold(shuffle=True, random_state=42)

            recall_scores, f1_scores, pr_auc_scores = [], [], []

            for train_index, test_index in skf.split(features, target):
                x_train, x_test = features.iloc[train_index].to_numpy(), features.iloc[test_index].to_numpy()
                y_train, y_test = target.iloc[train_index].to_list(), target.iloc[test_index].to_list()

                self.model.fit(x_train, y_train)

                y_pred = self.model.predict(x_test)
                y_proba = self.model.predict_proba(x_test)

                recall_scores.append(recall_score(y_test, y_pred))
                f1_scores.append(f1_score(y_test, y_pred))
                precision, recall_pr, _ = precision_recall_curve(y_test, y_proba[:, 1])
                pr_auc_scores.append(auc(recall_pr, precision))

            self.logger.info("Создание и запись PR-Curve")

            temp_dir = project_root / "temp_results"
            temp_dir.mkdir(exist_ok=True)

            fig_path = temp_dir / f"pr_curve_{self.timestamp}.html"
            fig_pr_curve = visualization.log_pr_curve(y_test, y_proba)
            fig_pr_curve.write_html(fig_path)
            mlflow_utils.log_artifact(str(fig_path), "pr_curve.html")

            fig_path.unlink()
            shutil.rmtree(temp_dir)

            mlflow_utils.log_model(self.model)
            mlflow_utils.log_params(self.best_params)

            final_metrics = {
                "Recall": np.mean(recall_scores),
                "F1-score": np.mean(f1_scores),
                "PR-AUC": np.mean(pr_auc_scores),
                "Training_time": time.time() - start_time
            }
            mlflow_utils.log_metrics(final_metrics)

            self.logger.info(
                "Обучение завершено\n"
                f"Total time: {final_metrics['Training_time']:.2f}s\n"
                f"Final metrics:\n"
                f"Recall: {final_metrics['Recall']:.4f}\n"
                f"F1: {final_metrics['F1-score']:.4f}\n"
                f"PR-AUC: {final_metrics['PR-AUC']:.4f}"
            )

    def predict(self, x_test):
        return self.model.predict(x_test)

    def save_model(self):
        save_start = time.time()
        super().save_model(self.model_name, self.model)
        self.logger.info(f"Модель сохранена как {self.model_name} в {time.time() - save_start:.2f}s")

    def load_model(self):
        pass