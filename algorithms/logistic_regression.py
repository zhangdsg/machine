import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, epochs=500, batch_size=256):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.n_classes = None
        self.loss_history = []

    def _softmax(self, z):
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y_onehot):
        n_samples, n_features = X.shape
        self.n_classes = y_onehot.shape[1]
        self.weights = np.random.randn(n_features, self.n_classes) * 0.01
        self.bias = np.zeros((1, self.n_classes))
        self.loss_history = []

        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y_onehot[indices]

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                logits = X_batch @ self.weights + self.bias
                probs = self._softmax(logits)

                dw = (1 / len(X_batch)) * (X_batch.T @ (probs - y_batch))
                db = (1 / len(X_batch)) * np.sum(probs - y_batch, axis=0, keepdims=True)

                self.weights -= self.lr * dw
                self.bias -= self.lr * db

            full_logits = X @ self.weights + self.bias
            full_probs = self._softmax(full_logits)
            loss = -np.mean(np.sum(y_onehot * np.log(full_probs + 1e-8), axis=1))
            self.loss_history.append(loss)

            if epoch % 100 == 0:
                pass

    def predict(self, X):
        logits = X @ self.weights + self.bias
        probs = self._softmax(logits)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        logits = X @ self.weights + self.bias
        return self._softmax(logits)
