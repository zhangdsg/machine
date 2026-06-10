from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_knn(X_train, y_train, model_save_path, n_neighbors=5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_scaled, y_train)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump((model, scaler), model_save_path)
    return model, scaler

def predict_knn(model, scaler, X_test):
    X_scaled = scaler.transform(X_test)
    return model.predict(X_scaled)