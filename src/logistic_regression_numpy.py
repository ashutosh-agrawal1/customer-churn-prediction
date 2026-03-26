import numpy as np

class LogisticRegression:
    def __init__(self, iterations, learning_rate):
        self.iterations = iterations
        self.lr = learning_rate
        self.weight = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)  # numerical stability
        return 1 / (1 + np.exp(-z))

    def _loss(self, y, y_hat):
        epsilon = 1e-9
        return -np.mean(
            y * np.log(y_hat + epsilon) +
            (1 - y) * np.log(1 - y_hat + epsilon)
        )

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            linear_model = np.dot(X, self.weight) + self.bias
            y_hat = self._sigmoid(linear_model)

            dw = np.dot(X.T, (y_hat - y)) / n_samples
            db = np.sum(y_hat - y) / n_samples

            self.weight -= self.lr * dw
            self.bias -= self.lr * db

            loss = self._loss(y, y_hat)
            self.loss_history.append(loss)

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weight) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)