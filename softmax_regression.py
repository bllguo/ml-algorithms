import numpy as np
from sklearn.datasets import load_iris
import sklearn.linear_model


class SoftmaxRegression:
    def __init__(self, learning_rate=.001, max_iter=10000, tol=1e-9, regularizer=1):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.regularizer = regularizer

    def fit(self, x, y):
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape((-1, 1))
        y = np.array(y)
        x = np.concatenate((np.ones([x.shape[0], 1]), x), axis=1)
        weights = np.random.randn(len(np.unique(y)), x.shape[1])
        costs = np.zeros(self.max_iter)
        prev_cost = np.finfo('float64').max
        for i in range(self.max_iter):
            cost, grad = self.compute_cost(x, y, weights, self.regularizer)
            weights = weights - self.learning_rate * grad
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
        return np.argmax(np.concatenate((np.ones([test.shape[0], 1]), test), axis=1) @ self.weights.T, axis=1)

    def compute_cost(self, x, y, weights, regularizer):
        scores = x @ weights.T
        # computational trick
        scores = np.exp(scores - np.max(scores))
        class_scores = scores[range(x.shape[0]), y]
        loss = -np.sum(np.log(class_scores/np.sum(scores, axis=1))) / x.shape[0] + regularizer*np.sum(weights*weights)/2
        grad = (scores.T / np.sum(scores, axis=1)).T
        grad[range(x.shape[0]), y] -= 1
        grad = (grad.T @ x) / x.shape[0] + regularizer*weights
        return loss, grad


# Test against sklearn
x, y = load_iris(True)

lr = SoftmaxRegression(max_iter=10000, regularizer=0)
lr.fit(x, y)
print(lr.weights)
np.round(lr.predict(x))

sk = sklearn.linear_model.LogisticRegression(C=1e9)
sk.fit(x, y)
print(sk.intercept_)
print(sk.coef_)
sk.predict(x)

np.round(lr.predict(x)) == sk.predict(x)
