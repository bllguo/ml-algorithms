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
    def fit(self, x, y):
        """
        Fits model given X, y

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
        classes = pd.unique(y)
        self.x = x
        self.y = y
        self.parameters = {x: pd.DataFrame(columns=['mean', 'sd']) for x in classes}
        for c in classes:
            subset = x[y == c]
            self.parameters[c] = self.parameters[c].append([{'mean': subset[column].mean(),
                                                             'sd': subset[column].std()} for column in subset])
        return self
    
    def predict(self, test):
        """
        Predicts class labels given test data newX

        Parameters
        ----------
        test: array-like
            test data
        
        Returns
        -------
        C: Series
            Predicted labels for test data
        """
        classes = pd.unique(self.y)
        return test.apply(self.compute_class, axis=1, args=(self.parameters, classes))
    
    @staticmethod
    def compute_likelihood(row, params):
        """
        Computes likelihood for given observation rowX and Gaussian parameters params for a specific class

        Parameters
        ----------
        row: array-like, shape(n,)
            test observation
        params: Dataframe
            contains Gaussian parameters, s.t. rows are features and columns are mean and sd, respectively

        Returns
        -------
        L: log likelihood that rowX belongs to the class described by params
        """
        ll = 0.0
        for i in range(row.shape[0]):
            ll += np.log(norm(params.iloc[i, 0], params.iloc[i, 1]).pdf(row[i]))
        return ll

    def compute_class(self, row, parameters, classes):
        """
        Assigns class label for given observation rowX, parameters dict, and class labels classes, by choosing class with max likelihood

        Parameters
        ----------
        row: array-like, shape(n,)
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
        return(classes[np.argmax([self.compute_likelihood(row, params) for (_, params) in parameters.items()])])


# Test against sklearn
df = sns.load_dataset('iris')
y = df['species']
X = df.iloc[:, :4]

gnb = GaussianNaiveBayes()
gnb.fit(X, y)
gnbpred = gnb.predict(X).values

nb = sklearn.naive_bayes.GaussianNB()
nb.fit(X,y)
nbpred = nb.predict(X)
