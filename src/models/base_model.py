import sys
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from pipeline import feature_engineering_bib
from utils.paths import project_root

data_file = str(project_root) + "/data/raw/heart.csv"
data = pd.read_csv(data_file, encoding='utf-8')
df = feature_engineering_bib.label_encoder(data)