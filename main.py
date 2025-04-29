import os
import sys
import logging
from src.analysis import firstlook_analysis
from src.utils import paths, set_logger
from src.data_processing import download_data
from src.pipeline import train

if __name__ == '__main__':
    project_root = str(paths.project_root)
    
    os.makedirs('logs', exist_ok=True)
    set_logger.setup_logging(project_root, "main")

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

    print("Нужен ли первичный осмотр данных?[Y/n]")
    flag = input()

    if flag == 'Y':
        logger.info("Запущен первичный осмотр данных")
        checker = firstlook_analysis.DataChecker(data_file, project_root)
        checker.check_data()
        checker.generate_graphs()

    logger.info("Запущен pipeline обучения")
    train.train(data_file)

    logger.info("Конец работы")