import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_clean_titanic(train_path, test_path):
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)

    useful_cols = ['Passengerid', 'Age', 'Fare', 'Sex', 'sibsp', 'Parch', 'Pclass', 'Embarked']

    train_cols = useful_cols + ['2urvived']
    test_cols = useful_cols

    train = train_raw[[c for c in train_cols if c in train_raw.columns]].copy()
    test = test_raw[[c for c in test_cols if c in test_raw.columns]].copy()

    if '2urvived' in train.columns:
        train.rename(columns={'2urvived': 'Survived'}, inplace=True)

    train['Age'].fillna(train['Age'].median(), inplace=True)
    test['Age'].fillna(test['Age'].median(), inplace=True)
    train['Fare'].fillna(train['Fare'].median(), inplace=True)
    test['Fare'].fillna(test['Fare'].median(), inplace=True)

    if 'Embarked' in train.columns:
        train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
    if 'Embarked' in test.columns:
        test['Embarked'].fillna(test['Embarked'].mode()[0], inplace=True)

    le_sex = LabelEncoder()
    train['Sex'] = le_sex.fit_transform(train['Sex'])
    test['Sex'] = le_sex.transform(test['Sex'])

    if 'Embarked' in train.columns:
        le_emb = LabelEncoder()
        train['Embarked'] = le_emb.fit_transform(train['Embarked'])
        test['Embarked'] = le_emb.transform(test['Embarked'])
    else:
        train['Embarked'] = 0
        test['Embarked'] = 0

    X_train = train.drop(['Survived', 'Passengerid'], axis=1)
    y_train = train['Survived']
    X_test = test.drop(['Passengerid'], axis=1, errors='ignore')

    cols_order = sorted(X_train.columns)
    X_train = X_train[cols_order]
    X_test = X_test[cols_order]

    return X_train, y_train, X_test