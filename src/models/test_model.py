from sklearn.metrics import accuracy_score

import pandas as pd

def model_val_accuracy(model, target_train: pd.Series, target_val: pd.Series,
                       X_train: pd.DataFrame, X_val: pd.DataFrame):
    tr_score = accuracy_score(y_true=target_train, y_pred=model.predict(X_train))
    val_score = accuracy_score(y_true=target_val, y_pred=model.predict(X_val))
    return [tr_score, val_score]


def model_accuracy(model, target_test: pd.Series, X_test: pd.DataFrame):
    return accuracy_score(y_true=target_test, y_pred=model.predict(X_test))