import logging
from src.data_processing import data_preprocessing
from src.models import (
    catboost_clf
    # skl_logreg_clf,
    # skl_randforest_clf,
    # xgboost_clf
)

def train(data_file):
    df, splits = data_preprocessing.preprocessing(data_file)

    model = catboost_clf.CatBoostClf(df, splits)
    model.fit()
    model.save_model()