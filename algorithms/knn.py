import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=5, metric='euclidean'):
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _distance(self, a, b):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((a - b) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(a - b))
        return np.sqrt(np.sum((a - b) ** 2))

    def predict(self, X):
        X = np.array(X)
        predictions = []
        for x in X:
            dists = [self._distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(dists)[:self.k]
            k_labels = [self.y_train[i] for i in k_indices]
            most_common = Counter(k_labels).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)
