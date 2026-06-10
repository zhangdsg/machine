import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=10, min_samples_leaf=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.n_classes = None

    def _gini(self, y):
        classes = Counter(y)
        impurity = 1.0
        for count in classes.values():
            p = count / len(y)
            impurity -= p ** 2
        return impurity

    def _split(self, X_column, threshold):
        left = X_column <= threshold
        right = X_column > threshold
        return left, right

    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        parent_gini = self._gini(y)

        for feature in range(X.shape[1]):
            X_col = X[:, feature]
            thresholds = np.percentile(X_col, np.linspace(10, 90, 9))

            for threshold in thresholds:
                left, right = self._split(X_col, threshold)
                if np.sum(left) < self.min_samples_leaf or np.sum(right) < self.min_samples_leaf:
                    continue

                n_left, n_right = np.sum(left), np.sum(right)
                n_total = n_left + n_right
                gain = parent_gini - (n_left / n_total) * self._gini(y[left]) - (n_right / n_total) * self._gini(y[right])

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(set(y))

        if depth >= self.max_depth or n_samples < self.min_samples_split or n_classes == 1:
            most_common = Counter(y).most_common(1)[0][0]
            return Node(value=most_common)

        feature, threshold = self._best_split(X, y)
        if feature is None:
            most_common = Counter(y).most_common(1)[0][0]
            return Node(value=most_common)

        left, right = self._split(X[:, feature], threshold)
        left_node = self._build_tree(X[left], y[left], depth + 1)
        right_node = self._build_tree(X[right], y[right], depth + 1)
        return Node(feature=feature, threshold=threshold, left=left_node, right=right_node)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        self.root = self._build_tree(np.array(X), np.array(y))

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._traverse(x, self.root) for x in X])
