import numpy as np
from sklearn.datasets import load_iris
import sklearn.linear_model


class LogisticRegression:
    def __init__(self, weights=None, max_iter=100):
        self.weights = weights
        self.max_iter = max_iter

    def fit(self, x, y):
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape((-1, 1))
        y = np.array(y)
        x = np.concatenate((np.ones([x.shape[0], 1]), x), axis=1)
        y = y.reshape((-1, 1))
        for i in range(self.max_iter):
            self.iterative_reweighted_least_squares(x, y)
        self.x = x
        self.y = y
        return self

    def predict(self, test):
        test = np.array(test)
        if test.ndim == 1:
            test = test.reshape((-1, 1))
        test = np.concatenate((np.ones([test.shape[0], 1]), test), axis=1)
        return self.sigmoid(test @ self.weights.T).reshape(-1)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def iterative_reweighted_least_squares(self, x, y):
        if self.weights is None:
            weights = np.ones([1, x.shape[1]])/100
        else:
            weights = self.weights
        cross_entropy_gradient = x.T @ (self.sigmoid(x @ weights.T) - y)
        weighting_matrix = np.diag((self.sigmoid(x @ weights.T) * (1 - self.sigmoid(x @ weights.T))).reshape(-1))
        cross_entropy_hessian = x.T @ weighting_matrix @ x
        self.weights = (weights.T - np.linalg.inv(cross_entropy_hessian) @ cross_entropy_gradient).T
        print(self.weights)


# Test against sklearn
x, y = load_iris(True)
y = y[0:100]
x = x[0:100]

lr = LogisticRegression(max_iter=100)
lr.fit(x, y)
print(lr.weights)
np.round(lr.predict(x))

sk = sklearn.linear_model.LogisticRegression(C=1e9)
sk.fit(x, y)
print(sk.intercept_)
print(sk.coef_)
sk.predict(x)

np.round(lr.predict(x)) == sk.predict(x)