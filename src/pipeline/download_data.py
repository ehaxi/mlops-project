import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.paths import project_root

# Устанавливаем переменную окружения для Kaggle API используя абсолютный путь
kaggle_config_path = str(project_root) + '/config'
os.environ['KAGGLE_CONFIG_DIR'] = kaggle_config_path

from kaggle.api.kaggle_api_extended import KaggleApi

# Проверяем путь до файла
print(os.environ.get('KAGGLE_CONFIG_DIR'))

# Убедимся, что файл существует
config_path = os.environ.get('KAGGLE_CONFIG_DIR', '')
if not os.path.exists(os.path.join(config_path, 'kaggle.json')):
    raise FileNotFoundError("Файл kaggle.json не найден в указанном пути.")

# Создаем объект API
api = KaggleApi()
api.authenticate()  # Аутентификация через kaggle.json

# Задаем параметры для скачивания данных
dataset = 'fedesoriano/heart-failure-prediction'
download_path = project_root / 'data' / 'raw'

# Скачиваем датасет
api.dataset_download_files(dataset, path=download_path, unzip=True)

print(f"Dataset {dataset} has been downloaded to {download_path}")