import numpy as np

class SingleLayerNN:
    def __init__(self, input_dim, output_dim, learning_rate=0.1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.weights = np.random.randn(input_dim, output_dim)
        self.bias = np.zeros(output_dim)

    def predict(self, X):
        net = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(net)
        return y_pred

    def fit(self, X, y, num_epochs):
        for i in range(num_epochs):
            y_pred = self.predict(X)
            error = y - y_pred
            d_weights = np.dot(X.T, error * y_pred * (1 - y_pred))
            d_bias = np.sum(error * y_pred * (1 - y_pred), axis=0)
            self.weights += self.learning_rate * d_weights
            self.bias += self.learning_rate * d_bias

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
