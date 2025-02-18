from datetime import datetime
from random import random

import joblib

def save_model(model, path: str):
    path = f"{path}/fraud_model_v{datetime.now().date()}_{random()}.pkl"
    joblib.dump(model, path)

def load_model(path_file):
    return joblib.load(path_file)
