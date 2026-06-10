import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

from preprocess.titanic_preprocess import load_and_clean_titanic
from preprocess.housing_preprocess import load_and_split_housing
from preprocess.hog_feature import extract_hog_features
from models.linear_regression import train_housing, evaluate_housing
from models.knn import train_knn
from models.logistic_regression import train_logistic
from models.svm import train_svm

# ------------------- 图像加载函数 -------------------
def load_mnist_from_folder(data_root):
    X_train, y_train = [], []
    X_test, y_test = [], []
    for split in ['train', 'test']:
        split_path = os.path.join(data_root, split)
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"找不到 {split_path}，请检查 MNIST 文件夹结构")
        for label in range(10):
            class_path = os.path.join(split_path, str(label))
            if not os.path.isdir(class_path):
                continue
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    if split == 'train':
                        X_train.append(img)
                        y_train.append(label)
                    else:
                        X_test.append(img)
                        y_test.append(label)
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def load_cifar10_from_folder(data_root, label_map):
    X_train, y_train = [], []
    X_test, y_test = [], []
    for split in ['train', 'test']:
        split_path = os.path.join(data_root, split)
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"找不到 {split_path}，请检查 CIFAR-10 文件夹结构")
        for class_name, label in label_map.items():
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):
                continue
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if split == 'train':
                        X_train.append(img)
                        y_train.append(label)
                    else:
                        X_test.append(img)
                        y_test.append(label)
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

# ------------------- 主函数 -------------------
def main():
    os.makedirs('results', exist_ok=True)

    # a. 线性回归房价
    print("\n=== a. 线性回归房价预测 ===")
    X_train_h, X_test_h, y_train_h, y_test_h = load_and_split_housing('data/house_data.csv')
    model_lr = train_housing(X_train_h, y_train_h, 'models/housing_lr.pkl')
    mse = evaluate_housing(model_lr, X_test_h, y_test_h)
    with open('results/housing_linear_regression_mse.txt', 'w') as f:
        f.write(f"MSE: {mse}\n")
    print(f"房价预测 MSE: {mse}")

    # b. KNN Titanic
    print("\n=== b. KNN Titanic 二分类 ===")
    X_titanic, y_titanic, _ = load_and_clean_titanic('data/titanic_train_knn.csv', 'data/titanic_test_knn.csv')
    X_train_t, X_val_t, y_train_t, y_val_t = train_test_split(X_titanic, y_titanic, test_size=0.2, random_state=42)
    train_knn(X_train_t, y_train_t, 'models/knn_titanic.pkl', n_neighbors=5)
    model_knn, scaler_knn = joblib.load('models/knn_titanic.pkl')
    X_val_scaled = scaler_knn.transform(X_val_t)
    acc_knn_titanic = accuracy_score(y_val_t, model_knn.predict(X_val_scaled))
    with open('results/knn_titanic_accuracy.txt', 'w') as f:
        f.write(f"Accuracy: {acc_knn_titanic}\n")
    print(f"KNN Titanic 准确率: {acc_knn_titanic}")

    # b. KNN MNIST 十分类
    print("\n=== b. KNN MNIST 十分类 ===")
    X_train_m, y_train_m, X_test_m, y_test_m = load_mnist_from_folder('data/mnist_split')
    sample_train = 500
    sample_test = 200
    X_train_sampled, y_train_sampled = [], []
    for c in range(10):
        idx = np.where(y_train_m == c)[0][:sample_train]
        X_train_sampled.extend(X_train_m[idx])
        y_train_sampled.extend(y_train_m[idx])
    X_train_sampled = np.array(X_train_sampled)
    y_train_sampled = np.array(y_train_sampled)
    X_test_sampled, y_test_sampled = [], []
    for c in range(10):
        idx = np.where(y_test_m == c)[0][:sample_test]
        X_test_sampled.extend(X_test_m[idx])
        y_test_sampled.extend(y_test_m[idx])
    X_test_sampled = np.array(X_test_sampled)
    y_test_sampled = np.array(y_test_sampled)

    X_train_flat = X_train_sampled.reshape(len(X_train_sampled), -1)
    X_test_flat = X_test_sampled.reshape(len(X_test_sampled), -1)
    train_knn(X_train_flat, y_train_sampled, 'models/knn_mnist.pkl', n_neighbors=3)
    model_knn_mnist, scaler_mnist = joblib.load('models/knn_mnist.pkl')
    X_test_scaled = scaler_mnist.transform(X_test_flat)
    acc_knn_mnist = accuracy_score(y_test_sampled, model_knn_mnist.predict(X_test_scaled))
    with open('results/knn_mnist_accuracy.txt', 'w') as f:
        f.write(f"Accuracy: {acc_knn_mnist}\n")
    print(f"KNN MNIST 准确率: {acc_knn_mnist}")

    # b. KNN CIFAR-10 十分类
    print("\n=== b. KNN CIFAR-10 十分类 ===")
    label_map_cifar = {
        'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4,
        'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9
    }
    X_train_c, y_train_c, X_test_c, y_test_c = load_cifar10_from_folder('data/cifar10_images', label_map_cifar)
    sample_train_c = 200
    sample_test_c = 50
    X_train_sampled_c, y_train_sampled_c = [], []
    for c in range(10):
        idx = np.where(y_train_c == c)[0][:sample_train_c]
        X_train_sampled_c.extend(X_train_c[idx])
        y_train_sampled_c.extend(y_train_c[idx])
    X_train_sampled_c = np.array(X_train_sampled_c)
    y_train_sampled_c = np.array(y_train_sampled_c)
    X_test_sampled_c, y_test_sampled_c = [], []
    for c in range(10):
        idx = np.where(y_test_c == c)[0][:sample_test_c]
        X_test_sampled_c.extend(X_test_c[idx])
        y_test_sampled_c.extend(y_test_c[idx])
    X_test_sampled_c = np.array(X_test_sampled_c)
    y_test_sampled_c = np.array(y_test_sampled_c)

    def to_gray_flatten(images):
        gray = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in images])
        return gray.reshape(len(gray), -1)

    X_train_gray = to_gray_flatten(X_train_sampled_c)
    X_test_gray = to_gray_flatten(X_test_sampled_c)
    train_knn(X_train_gray, y_train_sampled_c, 'models/knn_cifar10.pkl', n_neighbors=3)
    model_knn_cifar, scaler_cifar = joblib.load('models/knn_cifar10.pkl')
    X_test_scaled_c = scaler_cifar.transform(X_test_gray)
    acc_knn_cifar = accuracy_score(y_test_sampled_c, model_knn_cifar.predict(X_test_scaled_c))
    with open('results/knn_cifar10_accuracy.txt', 'w') as f:
        f.write(f"Accuracy: {acc_knn_cifar}\n")
    print(f"KNN CIFAR-10 准确率: {acc_knn_cifar}")

    # c. 逻辑回归 Titanic
    print("\n=== c. 逻辑回归 Titanic 二分类 ===")
    train_logistic(X_train_t, y_train_t, 'models/logistic_titanic.pkl')
    model_log, scaler_log = joblib.load('models/logistic_titanic.pkl')
    X_val_scaled_log = scaler_log.transform(X_val_t)
    acc_log_titanic = accuracy_score(y_val_t, model_log.predict(X_val_scaled_log))
    with open('results/logistic_titanic_accuracy.txt', 'w') as f:
        f.write(f"Accuracy: {acc_log_titanic}\n")
    print(f"逻辑回归 Titanic 准确率: {acc_log_titanic}")

    # c. 逻辑回归 + HOG (MNIST 4 vs 9)
    print("\n=== c. 逻辑回归 + HOG: MNIST 4 vs 9 ===")
    X_mnist_all, y_mnist_all, _, _ = load_mnist_from_folder('data/mnist_split')
    idx_49 = np.where((y_mnist_all == 4) | (y_mnist_all == 9))[0]
    X_mnist_49 = X_mnist_all[idx_49]
    y_mnist_49 = y_mnist_all[idx_49]
    X_hog_49 = extract_hog_features(X_mnist_49, resize=(28,28))
    X_train_hog, X_test_hog, y_train_hog, y_test_hog = train_test_split(X_hog_49, y_mnist_49, test_size=0.2, random_state=42)
    train_logistic(X_train_hog, y_train_hog, 'models/logistic_mnist_49_hog.pkl')
    model_log_hog, scaler_log_hog = joblib.load('models/logistic_mnist_49_hog.pkl')
    X_test_hog_scaled = scaler_log_hog.transform(X_test_hog)
    acc_log_mnist_hog = accuracy_score(y_test_hog, model_log_hog.predict(X_test_hog_scaled))
    with open('results/logistic_mnist_4vs9_hog.txt', 'w') as f:
        f.write(f"Accuracy: {acc_log_mnist_hog}\n")
    print(f"逻辑回归+HOG (MNIST 4vs9) 准确率: {acc_log_mnist_hog}")

    # c. 逻辑回归 + HOG (CIFAR-10 cat vs dog)
    print("\n=== c. 逻辑回归 + HOG: CIFAR-10 cat vs dog ===")
    X_cifar_all, y_cifar_all, _, _ = load_cifar10_from_folder('data/cifar10_images', label_map_cifar)
    idx_catdog = np.where((y_cifar_all == 3) | (y_cifar_all == 5))[0]
    X_cifar_cd = X_cifar_all[idx_catdog]
    y_cifar_cd = y_cifar_all[idx_catdog]
    X_cifar_gray = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in X_cifar_cd])
    X_hog_cd = extract_hog_features(X_cifar_gray, resize=(32,32))
    X_train_hog_c, X_test_hog_c, y_train_hog_c, y_test_hog_c = train_test_split(X_hog_cd, y_cifar_cd, test_size=0.2, random_state=42)
    train_logistic(X_train_hog_c, y_train_hog_c, 'models/logistic_cifar_catdog_hog.pkl')
    model_log_hog_c, scaler_log_hog_c = joblib.load('models/logistic_cifar_catdog_hog.pkl')
    X_test_hog_c_scaled = scaler_log_hog_c.transform(X_test_hog_c)
    acc_log_cifar_hog = accuracy_score(y_test_hog_c, model_log_hog_c.predict(X_test_hog_c_scaled))
    with open('results/logistic_cifar_catvsdog_hog.txt', 'w') as f:
        f.write(f"Accuracy: {acc_log_cifar_hog}\n")
    print(f"逻辑回归+HOG (CIFAR-10 cat vs dog) 准确率: {acc_log_cifar_hog}")

    # d. SVM Titanic
    print("\n=== d. SVM Titanic 二分类 ===")
    train_svm(X_train_t, y_train_t, 'models/svm_titanic_main.pkl')
    model_svm_t, scaler_svm_t = joblib.load('models/svm_titanic_main.pkl')
    X_val_svm_t = scaler_svm_t.transform(X_val_t)
    acc_svm_titanic = accuracy_score(y_val_t, model_svm_t.predict(X_val_svm_t))
    with open('results/svm_titanic_accuracy_main.txt', 'w') as f:
        f.write(f"Accuracy: {acc_svm_titanic}\n")
    print(f"SVM Titanic 准确率: {acc_svm_titanic}")

    # d. SVM + HOG (MNIST 4 vs 9)
    print("\n=== d. SVM + HOG: MNIST 4 vs 9 ===")
    train_svm(X_train_hog, y_train_hog, 'models/svm_mnist_49_hog.pkl')
    model_svm_hog, scaler_svm_hog = joblib.load('models/svm_mnist_49_hog.pkl')
    X_test_svm_hog_scaled = scaler_svm_hog.transform(X_test_hog)
    acc_svm_mnist_hog = accuracy_score(y_test_hog, model_svm_hog.predict(X_test_svm_hog_scaled))
    with open('results/svm_mnist_4vs9_hog.txt', 'w') as f:
        f.write(f"Accuracy: {acc_svm_mnist_hog}\n")
    print(f"SVM+HOG (MNIST 4vs9) 准确率: {acc_svm_mnist_hog}")

    # d. SVM + HOG (CIFAR-10 cat vs dog)
    print("\n=== d. SVM + HOG: CIFAR-10 cat vs dog ===")
    train_svm(X_train_hog_c, y_train_hog_c, 'models/svm_cifar_catdog_hog.pkl')
    model_svm_hog_c, scaler_svm_hog_c = joblib.load('models/svm_cifar_catdog_hog.pkl')
    X_test_svm_hog_c_scaled = scaler_svm_hog_c.transform(X_test_hog_c)
    acc_svm_cifar_hog = accuracy_score(y_test_hog_c, model_svm_hog_c.predict(X_test_svm_hog_c_scaled))
    with open('results/svm_cifar_catvsdog_hog.txt', 'w') as f:
        f.write(f"Accuracy: {acc_svm_cifar_hog}\n")
    print(f"SVM+HOG (CIFAR-10 cat vs dog) 准确率: {acc_svm_cifar_hog}")

    print("\n所有任务完成！结果保存在 results/ 文件夹中。")

if __name__ == '__main__':
    main()