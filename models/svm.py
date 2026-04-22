from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_svm(X_train, y_train, model_save_path, kernel='rbf', C=1.0):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = SVC(kernel=kernel, C=C, probability=True)
    model.fit(X_scaled, y_train)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump((model, scaler), model_save_path)
    return model, scaler