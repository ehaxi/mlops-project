import os
import subprocess
import platform

# Путь к корню проекта
project_root = os.path.dirname(os.path.abspath(__file__))

# Пути к хранилищам
backend_store_uri = os.path.join(project_root, "mlflow.db")
artifact_root = os.path.join(project_root, "logs", "mlflow_logs")

# Убедимся, что папка для артефактов существует
os.makedirs(artifact_root, exist_ok=True)

# Формируем URI для запуска
backend_store_uri_str = f"sqlite:///{backend_store_uri}"
artifact_root_uri_str = f"file:///{artifact_root.replace(os.sep, '/')}"  # Для Windows слэши
host = "127.0.0.1"
port = "5000"

# Команда запуска
cmd = [
    "mlflow", "server",
    "--backend-store-uri", backend_store_uri_str,
    "--default-artifact-root", artifact_root_uri_str,
    "--host", host,
    "--port", port
]

# Приведение к строке (особенно важно для Windows)
command_str = " ".join(cmd) if platform.system() != "Windows" else " ".join(cmd)

# print("🔧 Запуск MLflow Tracking Server:")
# print(f"📁 Backend URI: {backend_store_uri_str}")
# print(f"📁 Artifact Root: {artifact_root_uri_str}")
# print(f"🌐 URL: http://{host}:{port}\n")

subprocess.run(command_str, shell=True)