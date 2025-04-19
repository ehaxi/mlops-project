from pathlib import Path

# Получим путь до корневой папки проекта, чтобы не прописывать этот код каждый раз,
# когда мы хотим из одного файла получить доступ к файлу из другой папки
current_file = Path(__file__).resolve()
project_root = current_file.parents[2] 