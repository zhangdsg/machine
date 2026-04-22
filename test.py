from preprocess.titanic_preprocess import load_and_clean_titanic
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

if __name__ == '__main__':
    X_train_full, y_train_full, _ = load_and_clean_titanic(
        'data/titanic_train_knn.csv',
        'data/titanic_test_knn.csv'
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )
    model, scaler = joblib.load('models/svm_titanic.pkl')
    X_val_scaled = scaler.transform(X_val)
    acc = accuracy_score(y_val, model.predict(X_val_scaled))
    print(f"SVM Titanic 验证集准确率: {acc:.4f}")

    os.makedirs('results', exist_ok=True)
    with open('results/svm_titanic_accuracy.txt', 'w') as f:
        f.write(f"Accuracy: {acc}\n")