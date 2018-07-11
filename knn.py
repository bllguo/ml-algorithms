import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier


class KNearestNeighborsClassifier:
    """
    K-Nearest Neighbors classifier toy implementation, with Euclidean distance. Instantiate with empty constructor, then fit / predict.

    Attributes
    ----------
    X: array-like
        training data
    y: array-like
        training labels
    K: int
        number of nearest neighbors
    """
    def fit(self, x, y):
        """
        Initialize K-NN model

        Parameters
        ----------
        x: array-like
            training data
        y: array-like
            training labels

        Returns
        -------
        self: object
        """
        self.x = np.array(x)
        self.y = np.array(y)
        return self

    def predict(self, test, k=5):
        """
        Predicts class labels given test data newX, by looking at K nearest neighbors

        Parameters
        ----------
        test: array-like
            test data
        k: int
            number of neighbors

        Returns
        -------
        C: Series
            Predicted labels for test data
        """
        return np.apply_along_axis(self.compute_class, axis=1, arr=test, k=k)

    def compute_class(self, row, k):
        """
        Assigns class label for given observation rowX

        Parameters
        ----------
        row: array-like, shape(n,)
            test observation
        k: int
            number of neighbors

        Returns
        -------
        C: int/string
            Predicted class label for row
        """
        distances = np.apply_along_axis(np.linalg.norm, axis=1, arr=row-self.x)
        labels, counts = np.unique(self.y[np.argpartition(distances, k)[:k]], return_counts=True)
        return labels[np.argmax(counts)]


# Test against sklearn
X, y = load_iris(True)

knn = KNearestNeighborsClassifier()
knn.fit(X, y)
knnpred = knn.predict(X)

sk = KNeighborsClassifier()
sk.fit(X, y)
skpred = sk.predict(X)
