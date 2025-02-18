import pandas as pd

from sklearn.model_selection import GridSearchCV

def train_model(model, features: pd.DataFrame, target: pd.Series):
    return model.fit(features, target)

def fine_tuning_model(model, params, cv, X_train: pd.DataFrame, y_train: pd.Series):
    new_model = GridSearchCV(estimator=model, param_grid=params, cv=cv)
    new_model.fit(X_train, y_train)
    return new_model


