from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import os

def train_housing(X_train, y_train, model_save_path):
    model = LinearRegression()
    model.fit(X_train, y_train)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    return model

def evaluate_housing(model, X_test, y_test):
    pred = model.predict(X_test)
    return mean_squared_error(y_test, pred)