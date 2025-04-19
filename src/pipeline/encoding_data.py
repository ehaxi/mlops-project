import pandas as pd
from sklearn.preprocessing import LabelEncoder

def label_encoder(df: pd.DataFrame):
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    corr = df.corr()
    return corr