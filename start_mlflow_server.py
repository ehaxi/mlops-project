# import os
# import subprocess
# import platform
# import logging
# from pathlib import Path
# from src.utils.paths import project_root
# from src.utils import set_logger 

# logger = logging.getLogger(__name__)

# # Пути к хранилищам
# backend_store_uri = Path(project_root) / "mlflow_server/data_local"
# artifact_root = Path(project_root) / "mlflow_server/artifacts"

# # Убедимся, что папка для артефактов существует
# os.makedirs(artifact_root, exist_ok=True)

# # Формируем URI для запуска
# backend_store_uri_str = f"sqlite:///{backend_store_uri}"
# artifact_root_uri_str = f"file:///{artifact_root}"  
# host = "127.0.0.1"
# port = "5000"

# # Команда запуска
# cmd = [
#     "mlflow", "server",
#     "--backend-store-uri", backend_store_uri_str,
#     "--default-artifact-root", artifact_root_uri_str,
#     "--host", host,
#     "--port", port
# ]

# # Приведение к строке (особенно важно для Windows)
# command_str = " ".join(cmd)

# logger.info("Запуск MLflow Tracking Server:")
# logger.info(f"Backend URI: {backend_store_uri_str}")
# logger.info(f"Artifact Root: {artifact_root_uri_str}")
# logger.info(f"URL: http://{host}:{port}\n")

# subprocess.run(command_str, shell=False)


import os
import subprocess
import signal
import sys
import logging
from pathlib import Path
from src.utils.paths import project_root
from src.utils import set_logger 
    
os.makedirs('logs', exist_ok=True)
set_logger.setup_logging(str(project_root), "server")
logger = logging.getLogger(__name__)

def graceful_shutdown(signum, frame):
    logger.info("Получен сигнал завершения. Остановка MLflow сервера...")
    sys.exit(0)

def start_mlflow_server():
    # Пути к хранилищам
    backend_store_uri = Path(project_root) / "mlflow_server/data_local"
    artifact_root = Path(project_root) / "mlflow_server/artifacts"

    # Убедимся, что папки существуют
    os.makedirs(artifact_root, exist_ok=True)

    # Формируем URI для запуска
    backend_store_uri_str = f"sqlite:///{backend_store_uri}"
    artifact_root_uri_str = f"file:///{artifact_root}"  
    host = "127.0.0.1"
    port = "5000"

    # Команда запуска (как список аргументов)
    cmd = [
        "mlflow", "server",
        "--backend-store-uri", backend_store_uri_str,
        "--default-artifact-root", artifact_root_uri_str,
        "--host", host,
        "--port", port
    ]

    logger.info("Запуск MLflow Tracking Server:")
    logger.info(f"Backend URI: {backend_store_uri_str}")
    logger.info(f"Artifact Root: {artifact_root_uri_str}")
    logger.info(f"URL: http://{host}:{port}\n")

    try:
        # Регистрируем обработчики сигналов
        signal.signal(signal.SIGINT, graceful_shutdown)
        signal.signal(signal.SIGTERM, graceful_shutdown)
        
        # Запускаем процесс (не преобразовываем в строку!)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Ждем завершения процесса
        process.wait()
        
    except Exception as e:
        logger.error(f"Ошибка при запуске сервера: {str(e)}")
        sys.exit(1)
    finally:
        logger.info("Сервер MLflow остановлен")

if __name__ == "__main__":
    start_mlflow_server()