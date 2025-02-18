from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import pandas as pd

def data_preparation(data: pd.DataFrame, target: str):
    y = data[target]
    X = data.drop(columns=target, axis=1)

    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)

    scaler_impute = Pipeline(steps=[
       ("imputer", SimpleImputer(strategy="mean")),
       ("scaler", StandardScaler())
   ])
    X_tr_train = scaler_impute.fit_transform(X_train)
    X_tr_test = scaler_impute.transform(X_test)
    X_tr_val = scaler_impute.transform(X_val)

    return (X_tr_train, y_train), (X_tr_val, y_val), (X_tr_test, y_test)
