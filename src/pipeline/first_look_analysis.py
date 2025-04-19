import os
import sys
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Получение корня проекта
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.paths import project_root

# Класс для обработки данных
class DataProcessor:
    def __init__(self, data_file: str, plot_types=['histogram', 'boxplot', 'heatmap']):

        self.data_file = data_file
        self.data = None
        self.plot_types = plot_types

    def load_and_process_data(self):
        
        # Загружаем необработанный датасет heart.csv
        self.data = pd.read_csv(self.data_file, encoding='utf-8')

        # Изучаем стандартные числовые характеристики и смотрим на датасет
        print(self.data.head(10), '\n')
        print(self.data.dtypes, '\n')
        print("Описание качественных признаков датасета:", '\n', self.data.describe(include='object'), '\n')

        print("Уникальные значения типа object", '\n')
        for col in self.data.select_dtypes(include='object').columns:
            print(f"Столбец: {col}", ' ', self.data[col].unique())
        
        print('\n', "Описание количественных признаков датасета:", '\n', self.data.describe(include=['int64', 'float64']), '\n')

        # Проверка нужно ли проводить очистку
        print("Размеры датасета:", self.data.shape)
        print("Количество дубликатов:", (self.data.duplicated()).sum())
        print("Количество пропущенных значений:", '\n', (self.data.isnull()).sum(), '\n')



        # Можете добавить ещё что-либо на своё усмотрение или в зависимости от данных

    def generate_graphs(self):
        
        cols = self.data.columns

        # Если хотите использовать только количественные признаки, то удалите комментарий
        # и измените в блоке создания графиков cols на numerical_cols

        # numerical_cols = self.data.drop(columns=['FastingBS', 'HeartDisease'], inplace=False) \
        #     .select_dtypes(include=['int64', 'float64']).columns
        
        figures_path = str(project_root) + "/data/figures/"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Поддерживаемые типы графиков
        plot_functions = {
            'boxplot': lambda ax, col: sns.boxplot(y=self.data[col], ax=ax),
            'histogram': lambda ax, col: ax.hist(self.data[col], bins=10, alpha=0.7)
        }

        for plot_type in self.plot_types:
            # Проверим поддерживается ли текущий тип графика 
            if plot_type not in plot_functions and plot_type != 'heatmap':
                print(f"Тип графика {plot_type} не поддерживается.")
                continue

            if plot_type == 'heatmap':
                corr = self.data.corr()
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
                plt.title("Корреляционная матрица")
                plt.tight_layout()
                plt.savefig(os.path.join(figures_path, 'heatmap.jpeg'))
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
            filename = f"{plot_type}_{timestamp}.jpeg"
            plt.savefig(figures_path + filename)
            plt.close()


# Инициализация и запуск
data_file = str(project_root) + "/data/raw/heart.csv"

# Создаем объект и запускаем обработку данных
processor = DataProcessor(data_file)
processor.load_and_process_data()
processor.generate_graphs()