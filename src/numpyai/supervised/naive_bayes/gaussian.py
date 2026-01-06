"""Gaussian Naive Bayes classifier class."""

import numpy as np
from numpy.typing import NDArray, ArrayLike
from .base import BaseNB

class GaussianNB(BaseNB):
    """Naive Bayes classifier for classes that follow Gaussian distributions."""

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        """Constructs a new Gaussian Naive Bayes classifier.

        Parameters
        ----------
        var_smoothing : float, optional
            Portion of the largest variance of all features that is 
            added to variances for calculation stability, by default 1e-9
        """
        super().__init__()
        self.var_smoothing = var_smoothing

    def fit(self, x: ArrayLike, y: ArrayLike) -> None:
        # Ensures inputs are Numpy arrays
        x = np.asarray(x)
        y = np.asarray(y)
        self._binariser_used = len(y.shape) == 1

        # Binarises target labels if necessary
        if self._binariser_used:
            y = self._binariser.fit_transform(y)
            self._classes = self._binariser._classes
        else:
            self._classes = np.arange(y.shape[1])

        # Ensures labels are in one-hot encoded form (for binary classification)
        if y.shape[1] == 1:
            y = np.hstack([1 - y, y])

        # Stores number of input features and classse
        self._n_features_in = x.shape[1]
        n_classes = len(self._classes)

        self._theta = np.zeros((n_classes, self._n_features_in))
        self._var = np.zeros((n_classes, self._n_features_in))
        self._class_count = np.zeros(n_classes)
        for i in range(n_classes):
            mask = y[:, i].astype(bool)
            x_feature = x[mask]
            self._theta[i] = np.mean(x_feature, axis=0)
            self._var[i] = np.var(x_feature, axis=0)
            self._class_count[i] = sum(mask)

        self._class_prior = self._class_count / np.sum(self._class_count)
        self._class_log_prior = np.log(self._class_prior)

    def predict_log_proba(self, x: ArrayLike) -> NDArray:   
        # Ensures inputs are Numpy arrays
        x = np.asarray(x)

        # Calculates the probabilities for each class
        log_probabilities = np.zeros((len(x), len(self._classes)))
        for i in range(log_probabilities.shape[0]):
            sample = x[i]
            for j in range(log_probabilities.shape[1]):
                mean, var = self._theta[j], self._var[j]

                v1 = 2 * np.pi * var
                v1[v1 == 0] += self.var_smoothing
                var[var == 0] += self.var_smoothing

                log_likelihood = -0.5 * (np.log(v1) + (np.square(sample - mean) / var))
                log_probabilities[i, j] = np.sum(log_likelihood) + self._class_log_prior[j]

        return log_probabilities
