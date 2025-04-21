import os
import logging

from src.pipeline import (
    paths,
    firstlook_analysis,
    download_data,
    set_logger,
    data_preprocessing
)


if __name__ == '__main__':
    
    os.makedirs('logs', exist_ok=True)
    set_logger.setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("Начало работы")

    # При необходимости замените имя файла!
    file_path = "data/raw/heart.csv"

    if not os.path.exists(file_path):
        logger.info("Датасета не существует. Скачивание датасета с Kaggle")
        download_data.installer()
    else:
        logger.info("Датасет найден")

    data_file = str(paths.project_root) + "/data/raw/heart.csv"

    logger.info("Запущен первичный осмотр данных")
    checker = firstlook_analysis.DataChecker(data_file, paths.project_root)
    checker.check_data()
    checker.generate_graphs()

    logger.info("Запущен препроцессер данных")
    data_preprocessing.preprocessing(data_file) 

    logger.info("Конец работы")