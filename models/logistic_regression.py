from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_logistic(X_train, y_train, model_save_path, C=1.0):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_scaled, y_train)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump((model, scaler), model_save_path)
    return model, scaler