import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_housing(data_path, test_size=0.2, random_state=42):
    df = pd.read_csv(data_path)
    X = df[['x1', 'x2', 'x3', 'x4']]
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test