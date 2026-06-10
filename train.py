from preprocess.titanic_preprocess import load_and_clean_titanic
from models.svm import train_svm

if __name__ == '__main__':
    X_train, y_train, _ = load_and_clean_titanic(
        'data/titanic_train_knn.csv',
        'data/titanic_test_knn.csv'
    )
    model_save_path = 'models/svm_titanic.pkl'
    train_svm(X_train, y_train, model_save_path)
    print(f"SVM Titanic 模型已保存到 {model_save_path}")