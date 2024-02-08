import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from DimensionalReductionMethods import sLDA
import numpy as np

class Regression:
    """
    A class for performing regression on a set of features( Inputs) and labels (Outputs).

    Attributes:
        features (numpy.ndarray): An array of shape (n_samples, n_features) containing the input features.
        labels (numpy.ndarray): An array of shape (n_samples,) containing the target labels. 
    """

    def __init__(self, features, labels):
        """
        Initializes a new Regression object with the given input features and target labels.

        Args:
            features (numpy.ndarray): An array of shape (n_samples, n_features) containing the input features.
            labels (numpy.ndarray): An array of shape (n_samples,) containing the target labels.
        """
        self.features = features
        self.labels = labels

    def fit(self, method='PCA', maxModes=3, slices=20, alpha=0.8, nvar=3, fitInds=None):
        """
        Fits a regression model to the input features and target labels.

        Args:
            method (str): The regression method to use. Can be 'PCA' or 'LDA'.
            maxModes (int): The maximum number of modes to use for regression.
            slices (int): The number of slices to use for LDA.
            alpha (float): The regularization parameter to use for LDA.
            nvar (int): The number of variables to use for LDA.
            fitInds (numpy.ndarray): An array of indices to use for fitting the regression model.

        Returns:
            timeSeries (numpy.ndarray): An array of shape (n_samples, maxModes) containing the time series data.
            modes (numpy.ndarray): An array of shape (maxModes, n_features) containing the regression modes.
            fitInds (numpy.ndarray): An array of indices used for fitting the regression model.
        """
        if fitInds is None:
            self.fitInds = np.arange(1, self.features.shape[0], 2)
        else:
            self.fitInds = fitInds

        testInds = np.setdiff1d(np.arange(self.features.shape[0]), self.fitInds)
        features = self.features[self.fitInds, :]
        labels = self.labels[self.fitInds]
        print('fitinds', features)
        if len(self.fitInds) == 0:
            raise ValueError("fitInds array is empty")
        if method == 'PCA':
                _, s, modes = svds(features, k=maxModes)
                idx = np.argsort(-s)
                s, modes = s[idx], modes.take(idx, axis=0)
                timeSeries = features @ modes.T
                self.timeSeries, self.modes = timeSeries, modes
        elif method == 'LDA':
            data = np.copy(features.T)
            target = np.copy(labels)
            modes, L, _ = sLDA(data, target, slices, alpha, nvar)
            timeSeries = features @ modes
            self.timeSeries, self.modes = timeSeries, modes
      
            
        else:
            raise NotImplementedError
    ##              # Compute the regression variables for all modes
        regvars = np.hstack((self.timeSeries[:, :1+maxModes], np.ones((self.timeSeries.shape[0], 1))))
       # Compute c coefficients for all modes
        c = np.linalg.lstsq(regvars, labels, rcond=None)[0]

        self.method,self.maxModes,self.c,self.testInds =  method,maxModes,c,testInds
        return self.timeSeries, self.modes, self.fitInds
    def predict(self, features=None):
        """
        Predicts the target labels for a new set of input features.
    Returns:
            y_pred (numpy.ndarray): An array of shape (n_samples,) containing the predicted target labels.
        """
        if features is None:
            features = self.features

        if self.method == 'PCA':
            timeSeries = features @ self.modes.T
        elif self.method == 'LDA':
            timeSeries = features @ self.modes
        else:
            raise NotImplementedError

        regvars = np.hstack((timeSeries[:, :1+self.maxModes], np.ones((timeSeries.shape[0], 1))))
        y_pred = regvars @ self.c
        return y_pred
