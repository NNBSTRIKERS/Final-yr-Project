import numpy as np
from scipy.stats import multivariate_normal

class BayesianDiscriminantFunction:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.prior = np.zeros(num_classes)
        self.mean = np.zeros((num_classes, 2))
        self.covariance = np.zeros((num_classes, 2, 2))

    def fit(self, X, y):
        for i in range(self.num_classes):
            X_class = X[y == i]
            self.prior[i] = len(X_class) / len(X)
            self.mean[i] = np.mean(X_class, axis=0)
            self.covariance[i] = np.cov(X_class.T)

    def predict(self, X):
        y_pred = []
        for x in X:
            likelihood = np.zeros(self.num_classes)
            for i in range(self.num_classes):
                likelihood[i] = multivariate_normal.pdf(x, mean=self.mean[i], cov=self.covariance[i])
            posterior = likelihood * self.prior
            y_pred.append(np.argmax(posterior))
        return y_pred
