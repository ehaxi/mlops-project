import os
import logging
from pipeline import paths

def installer():

    logger = logging.getLogger(__name__)

    logger.info("Установка переменной окружения")
    # Устанавливаем переменную окружения для Kaggle API используя абсолютный путь
    kaggle_config_path = str(paths.project_root) + '/config'
    os.environ['KAGGLE_CONFIG_DIR'] = kaggle_config_path

    from kaggle.api.kaggle_api_extended import KaggleApi

    # Проверяем путь до файла
    logger.info(f"Провекрка пути: {os.environ.get('KAGGLE_CONFIG_DIR')}")

    logger.info("Проверка наличия API-ключа пользователя")
    # Убедимся, что файл существует
    config_path = os.environ.get('KAGGLE_CONFIG_DIR', '')
    if not os.path.exists(os.path.join(config_path, 'kaggle.json')):
        logger.critical("API-ключ не найден")
        raise FileNotFoundError("Файл kaggle.json не найден в указанном пути.")

    logger.info("Создание объекта API")
    # Создаем объект API
    api = KaggleApi()
    api.authenticate()  # Аутентификация через kaggle.json

    # Задаем параметры для скачивания данных
    dataset = 'fedesoriano/heart-failure-prediction'
    download_path = paths.project_root / 'data' / 'raw'

    logger.info("Скачивание датасета")
    # Скачиваем датасет
    api.dataset_download_files(dataset, path=download_path, unzip=True)

    logger.info(f"Датасет {dataset} успешно установлен в {download_path}")