import os
import sys
import logging
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from . import encoding_data

# Получение корня проекта
#sys.path.append(str(Path(__file__).resolve().parents[1]))

# Класс для обработки данных
class DataChecker:
    def __init__(self, data_file: str, project_root: Path, plot_types=['histogram', 'boxplot', 'heatmap']):

        self.data_file = data_file
        self.data = pd.read_csv(self.data_file, encoding='utf-8')
        self.project_root = project_root
        self.plot_types = plot_types
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = logging.getLogger(__name__)

    def check_data(self):

        self.logger.info("Начало работы firstlook_analysis.check_data")

        self.logger.info("Организация хранения данных о датасете")
        # Создаем подпапку в которой будут храниться данные и настраиваем файл для записи
        data_info_filename = f"data_info_{self.timestamp}.txt"
        data_path = str(self.project_root) + "/data/processed/"
        data_info_filepath = os.path.join(data_path, data_info_filename)

        self.logger.info("Запись данных в .txt")
        # Открываем файл для записи
        with open(data_info_filepath, 'w', encoding='utf-8') as info_file:
            # Записываем первые данные в файл
            info_file.write(str(self.data.head(10)) + '\n\n')
            info_file.write(str(self.data.dtypes) + '\n\n')

            info_file.write("Описание качественных признаков датасета:\n")
            info_file.write(str(self.data.describe(include='object')) + '\n\n')

            info_file.write("Уникальные значения типа object:\n")
            for col in self.data.select_dtypes(include='object').columns:
                info_file.write(f"Столбец: {col}  {self.data[col].unique()}\n")

            info_file.write("\nОписание количественных признаков датасета:\n")
            info_file.write(str(self.data.describe(include=['int64', 'float64'])) + '\n\n')

            # Проверка на очистку данных
            info_file.write("Размеры датасета: " + str(self.data.shape) + '\n')
            info_file.write("Количество дубликатов: " + str((self.data.duplicated()).sum()) + '\n')
            info_file.write("Количество пропущенных значений:\n" + str((self.data.isnull()).sum()) + '\n')

        self.logger.info("Данные записаны")

        # Можете добавить ещё что-либо на своё усмотрение или в зависимости от данных

    def generate_graphs(self):
        
        self.logger.info("Начало работы firstlook_analysis.generate_graphs")

        cols = self.data.columns

        # Если хотите использовать только количественные признаки, то удалите комментарий
        # и измените в блоке создания графиков cols на numerical_cols

        # numerical_cols = self.data.drop(columns=['FastingBS', 'HeartDisease'], inplace=False) \
        #     .select_dtypes(include=['int64', 'float64']).columns
        
        self.logger.info("Организация хранения графиков")
        # Создаем подпапку в которой будут храниться графики
        figures_path = os.path.join(str(self.project_root), "data", "figures", self.timestamp)
        os.makedirs(figures_path)

        # Поддерживаемые типы графиков
        plot_functions = {
            'boxplot': lambda ax, col: sns.boxplot(y=self.data[col], ax=ax),
            'histogram': lambda ax, col: ax.hist(self.data[col], bins=10, alpha=0.7)
        }

        # Чтобы heatmap работала корректно
        corr = encoding_data.label_encoder(self.data)

        self.logger.info("Начало создания графиков")
        for plot_type in self.plot_types:
            # Проверим поддерживается ли текущий тип графика 
            if plot_type not in plot_functions and plot_type != 'heatmap':
                print(f"Тип графика {plot_type} не поддерживается.")
                self.logger.warning(f"Тип графика {plot_type} не поддерживается.")
                continue

            if plot_type == 'heatmap':
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
                plt.title("Корреляционная матрица")
                plt.tight_layout()
                filename = f"{plot_type}_{self.timestamp}.jpeg"
                plt.savefig(os.path.join(figures_path, filename))
                plt.close()
                continue
            
            # Создание и сохранение графика
            n_plots = len(cols)
            plots_size = math.ceil(math.sqrt(n_plots))
            plt.figure(figsize=(5 * plots_size, 3 * plots_size))
            for i, col in enumerate(cols, 1):
                ax = plt.subplot(plots_size, n_plots // plots_size, i)
                plot_functions[plot_type](ax, col)
                ax.set_title(col)

            plt.tight_layout()
            filename = f"{plot_type}_{self.timestamp}.jpeg"
            plt.savefig(os.path.join(figures_path, filename))
            plt.close()

        self.logger.info("Графики сохранены")