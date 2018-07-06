import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import norm

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
    def fit(self, X, y):
        """
        Initialize K-NN model

        Parameters
        ----------
        X: array-like
            training data
        y: array-like
            training labels

        Returns
        -------
        self: object
        """
        self.X = X
        self.y = y
        return(self)
    
    def predict(self, newX, K=5):
        """
        Predicts class labels given test data newX, by looking at K nearest neighbors

        Parameters
        ----------
        newX: array-like
            test data
        K: int
            number of neighbors    
        
        Returns
        -------
        C: Series
            Predicted labels for test data
        """
        classes = pd.unique(self.y)
        return(newX.apply(self.computeClass, axis=1, args=(K,)))

    def computeClass(self, rowX, K):
        """
        Assigns class label for given observation rowX

        Parameters
        ----------
        rowX: array-like, shape(n,)
            test observation
        K: int
            number of neighbors

        Returns
        -------
        C: int/string
            Predicted class label for rowX
        """
        distances = (rowX-X).apply(np.linalg.norm, axis=1)
        labels, counts = np.unique(self.y[np.argpartition(distances, K)[:K]], return_counts=True)
        return(labels[np.argmax(counts)])

# Test against sklearn
df = sns.load_dataset('iris')
y = df['species']
X = df.iloc[:,:4]

knn = KNearestNeighborsClassifier()
knn.fit(X, y)
knnpred = knn.predict(X).values

sk = KNeighborsClassifier()
sk.fit(X,y)
skpred = sk.predict(X)