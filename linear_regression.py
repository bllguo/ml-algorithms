import numpy as np
from sklearn.datasets import load_iris
import sklearn.linear_model

class LinearRegression:
    def __init__(self, learning_rate=.001, max_iter=100000, tol=1e-9):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, x, y):
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape((-1, 1))
        y = np.array(y)
        x = np.concatenate((np.ones([x.shape[0], 1]), x), axis=1)
        y = y.reshape((-1, 1))
        weights = np.ones([1, x.shape[1]])
        costs = np.zeros(self.max_iter)
        prev_cost = np.finfo('float64').max
        for i in range(self.max_iter):
            weights = weights - (self.learning_rate / len(y)) * np.sum(x * (x @ weights.T - y), axis=0, keepdims=True)
            cost = self.compute_cost(x, y, weights.T)
            if prev_cost - cost < self.tol:
                break
            costs[i] = cost
            prev_cost = cost
        self.x = x
        self.y = y
        self.weights = weights
        self.costs = costs
        return self

    def predict(self, test):
        return np.concatenate((np.ones([test.shape[0], 1]), test), axis=1) @ self.weights

    @staticmethod
    def compute_cost(x, y, weights):
        return np.sum((x @ weights - y) ** 2) / (2.0 * len(y))


# Test against sklearn
x, y = load_iris(True)
y = x[:, 0]
x = x[:, 1:]

lr = LinearRegression(learning_rate=.001, max_iter=100000)
lr.fit(x, y)
print(lr.weights)

sk = sklearn.linear_model.LinearRegression()
sk.fit(x, y)
print(sk.intercept_)
print(sk.coef_)
