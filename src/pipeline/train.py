from src.data_processing import data_preprocessing
from src.models import catboost_clf

def train(data_file):
    df = data_preprocessing.preprocessing(data_file)

    model = catboost_clf.CatBoostClf(df)
    model.fit()
    model.save_model()