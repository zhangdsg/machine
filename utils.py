import numpy as np
import pandas as pd

DATA_DIR = "F:/MachineLearning/lasting learning/Dry_Bean_Dataset"
OUTPUT_DIR = "F:/MachineLearning/lasting learning/BeanClassifier/output"

CLASS_MAPPING = {
    'DERMASON': 'DERMASON', 'dermason': 'DERMASON', 'DERMASON ': 'DERMASON',
    'D3RMAS0N': 'DERMASON',
    'BARBUNYA': 'BARBUNYA', 'barbunya': 'BARBUNYA', 'BARBUNYA ': 'BARBUNYA',
    'HOROZ': 'HOROZ', 'horoz': 'HOROZ', 'HOROZ ': 'HOROZ', 'H0R0Z': 'HOROZ',
    'SEKER': 'SEKER', 'seker': 'SEKER', 'SEKER ': 'SEKER', 'S3K3R': 'SEKER',
    'SIRA': 'SIRA', 'sira': 'SIRA', 'SIRA ': 'SIRA',
    'CALI': 'CALI', 'cali': 'CALI', 'CALI ': 'CALI',
    'BOMBAY': 'BOMBAY', 'bombay': 'BOMBAY', 'BOMBAY ': 'BOMBAY', 'B0MBAY': 'BOMBAY',
}

FEATURE_COLS = [
    'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
    'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter',
    'Extent', 'Solidity', 'roundness', 'Compactness',
    'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'
]

def clean_dataframe(df):
    df = df.copy()
    for col in ['Perimeter', 'Solidity', 'Compactness']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in FEATURE_COLS:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    df['Class'] = df['Class'].str.strip().map(CLASS_MAPPING)
    df = df.dropna(subset=['Class'])
    return df

def standardize(X_train, X_val, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1
    X_train_s = (X_train - mean) / std
    X_val_s = (X_val - mean) / std
    X_test_s = (X_test - mean) / std
    return X_train_s, X_val_s, X_test_s, mean, std

def to_onehot(y, classes):
    mapping = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([mapping[val] for val in y])
    onehot = np.zeros((len(y_idx), len(classes)))
    onehot[np.arange(len(y_idx)), y_idx] = 1
    return onehot

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred, classes):
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    idx_map = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        cm[idx_map[t], idx_map[p]] += 1
    return cm
