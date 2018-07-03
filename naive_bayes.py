import pandas as pd
import numpy as np
import seaborn as sns
import sklearn.naive_bayes
from scipy.stats import norm

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes toy implementation. Instantiate with empty constructor, then fit / predict.

    Attributes
    ----------
    X: array-like
        training data
    y: array-like
        training labels
    parameters: dict
        keys are class labels, values are Dataframes containing MLE estimates of Gaussian parameters.
        Dataframes are built s.t. rows are features and columns are mean and sd, respectively
    """
    def fit(self, X, y):
        """
        Fits model given X, y

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
        classes = pd.unique(y)
        self.X = X
        self.y = y
        self.parameters = {x: pd.DataFrame(columns=['mean', 'sd']) for x in classes}
        for x in classes:
            subX = X[y == x]
            self.parameters[x] = self.parameters[x].append([{'mean': subX[column].mean(), 'sd': subX[column].std()} for column in subX])
        return self
    
    def predict(self, newX):
        """
        Predicts class labels given test data newX

        Parameters
        ----------
        newX: array-like
            test data
        
        Returns
        -------
        C: Series
            Predicted labels for test data
        """
        classes = pd.unique(self.y)
        return(newX.apply(self.computeClass, axis=1, args=(self.parameters, classes)))
    
    @staticmethod
    def computeLikelihood(rowX, params):
        """
        Computes likelihood for given observation rowX and Gaussian parameters params for a specific class

        Parameters
        ----------
        rowX: array-like, shape(n,)
            test observation
        params: Dataframe
            contains Gaussian parameters, s.t. rows are features and columns are mean and sd, respectively

        Returns
        -------
        L: log likelihood that rowX belongs to the class described by params
        """
        ll = 0.0
        for i in range(rowX.shape[0]):
            ll += np.log(norm(params.iloc[i, 0], params.iloc[i, 1]).pdf(rowX[i]))
        return(ll)

    def computeClass(self, rowX, parameters, classes):
        """
        Assigns class label for given observation rowX, parameters dict, and class labels classes, by choosing class with max likelihood

        Parameters
        ----------
        rowX: array-like, shape(n,)
            test observation
        parameters: dict
            keys are class labels, values are Dataframes containing MLE estimates of Gaussian parameters.
            Dataframes are built s.t. rows are features and columns are mean and sd, respectively
        classes: array
            class labels

        Returns
        -------
        C: int/string
            Predicted class label for rowX
        """
        return(classes[np.argmax([self.computeLikelihood(rowX, params) for (_, params) in parameters.items()])])

# Test against sklearn
df = sns.load_dataset('iris')
y = df['species']
X = df.iloc[:,:4]

gnb = GaussianNaiveBayes()
gnb.fit(X, y)
gnbpred = gnb.predict(X).values

nb = sklearn.naive_bayes.GaussianNB()
nb.fit(X,y)
nbpred = nb.predict(X)