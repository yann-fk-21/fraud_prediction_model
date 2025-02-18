from datetime import datetime
from random import random

import pandas as pd
import matplotlib.pyplot as plt

def save_csv(df: pd.DataFrame, path: str):
    path = f"{path}/fraud_data_v{datetime.now().date()}_{random()}.csv"
    df.to_csv(path, index=False)

def save_fig(path: str):
    plt.savefig(path)


