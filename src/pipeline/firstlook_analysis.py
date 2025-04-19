import os
import sys
import logging
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from encoding_data import label_encoder

# Получение корня проекта
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.paths import project_root

# Класс для обработки данных
class DataProcessor:
    def __init__(self, data_file: str, plot_types=['histogram', 'boxplot', 'heatmap']):

        self.data_file = data_file
        self.data = pd.read_csv(self.data_file, encoding='utf-8')
        self.plot_types = plot_types
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def check_data(self):

        # Создаем подпапку в которой будут храниться данные и настраиваем логи
        data_path = str(project_root) + "/data/processed/"
        logging.basicConfig(
            filename=data_path + f'check_{self.timestamp}.log',
            level=logging.INFO,
            format='%(message)s',
            encoding='utf-8'
        )

        # Изучаем стандартные числовые характеристики и смотрим на датасет
        logging.info(self.data.head(15).to_string())
        logging.info('\n' + str(self.data.dtypes))

        logging.info("\nОписание качественных признаков датасета:\n" +
                    self.data.describe(include='object').to_string())

        logging.info("\nУникальные значения типа object\n")
        for col in self.data.select_dtypes(include='object').columns:
            logging.info(f"Столбец: {col}   {self.data[col].unique()}")

        logging.info("\nОписание количественных признаков датасета:\n" +
                    self.data.describe(include=['int64', 'float64']).to_string())

        # Проверка нужно ли проводить очистку
        logging.info("\nРазмеры датасета: " + str(self.data.shape))
        logging.info("Количество дубликатов: " + str(self.data.duplicated().sum()))
        logging.info("Количество пропущенных значений:\n" +
                    str(self.data.isnull().sum()))


        # Можете добавить ещё что-либо на своё усмотрение или в зависимости от данных

    def generate_graphs(self):
        
        cols = self.data.columns

        # Если хотите использовать только количественные признаки, то удалите комментарий
        # и измените в блоке создания графиков cols на numerical_cols

        # numerical_cols = self.data.drop(columns=['FastingBS', 'HeartDisease'], inplace=False) \
        #     .select_dtypes(include=['int64', 'float64']).columns
        
        # Создаем подпапку в которой будут храниться графики
        figures_path = os.path.join(str(project_root), "data", "figures", self.timestamp)
        os.makedirs(figures_path)

        # Поддерживаемые типы графиков
        plot_functions = {
            'boxplot': lambda ax, col: sns.boxplot(y=self.data[col], ax=ax),
            'histogram': lambda ax, col: ax.hist(self.data[col], bins=10, alpha=0.7)
        }

        # Чтобы heatmap работала корректно
        corr = label_encoder(self.data)

        for plot_type in self.plot_types:
            # Проверим поддерживается ли текущий тип графика 
            if plot_type not in plot_functions and plot_type != 'heatmap':
                print(f"Тип графика {plot_type} не поддерживается.")
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


# Инициализация и запуск
data_file = str(project_root) + "/data/raw/heart.csv"

# Создаем объект и запускаем обработку данных
processor = DataProcessor(data_file)
processor.check_data()
processor.generate_graphs()