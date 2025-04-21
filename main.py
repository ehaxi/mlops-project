import subprocess
import time
import requests
import os
import sys
import logging
from src.analysis import firstlook_analysis
from src.utils import paths, set_logger
from src.data_processing import download_data
from src.pipeline import train

def start_mlflow_server():
    try:
        # Проверка: уже работает ли сервер
        response = requests.get("http://127.0.0.1:5000")
        if response.status_code == 200:
            print("[INFO] MLflow server is already running.")
            return
    except requests.exceptions.ConnectionError:
        print("[INFO] MLflow server is not running. Starting now...")

    # Запуск сервера в фоне
    subprocess.Popen(
        [
            "mlflow", "server",
            "--backend-store-uri", "file://logs/mlflow_logs",
            "--default-artifact-root", "file://logs/mlflow_logs",
            "--host", "127.0.0.1",
            "--port", "5000"
        ],
        cwd=".",  # путь откуда запускать, "." = текущая папка
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Дать серверу немного времени на запуск
    time.sleep(5)
    # print("[INFO] MLflow server started.")

if __name__ == '__main__':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

    pid = start_mlflow_server()

    project_root = str(paths.project_root)
    
    os.makedirs('logs', exist_ok=True)
    set_logger.setup_logging(project_root)

    logger = logging.getLogger(__name__)
    logger.info("Начало работы")

    # При необходимости замените имя файла!
    file_path = "data/raw/heart.csv"

    if not os.path.exists(file_path):
        logger.info("Датасета не существует. Скачивание датасета с Kaggle")
        download_data.installer(project_root)
    else:
        logger.info("Датасет найден")

    data_file = project_root + "/data/raw/heart.csv"

    logger.info("Запущен первичный осмотр данных")
    checker = firstlook_analysis.DataChecker(data_file, project_root)
    checker.check_data()
    checker.generate_graphs()

    logger.info("Запущен pipeline обучения")
    train.train(data_file)

    logger.info("Конец работы")