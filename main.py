import os
import logging
from datetime import datetime

from src.utils.paths import project_root
from src.pipeline.firstlook_analysis import DataChecker
from src.pipeline import download_data

def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/log_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

if __name__ == '__main__':
    os.makedirs('logs', exist_ok=True)
    setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("Начало работы")

    # При необходимости замените имя файла!
    file_path = "data/raw/heart.csv"

    if not os.path.exists(file_path):
        logger.info("Датасета не существует. Скачивание датасета с Kaggle")
        download_data.installer()
    else:
        logger.info("Датасет найден")

    data_file = str(project_root) + "/data/raw/heart.csv"

    logger.info("Запущен первичный осмотр данных")
    checker = DataChecker(data_file, project_root)
    checker.check_data()
    checker.generate_graphs()

    logger.info("Конец работы")