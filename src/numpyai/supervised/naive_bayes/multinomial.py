"""Multinomial Naive Bayes classifier class."""

import numpy as np
from numpy.typing import NDArray, ArrayLike
from .base import BaseNB

class MultinomialNB(BaseNB):
    """
    Naive Bayes classifier for multinomial models.
    
    Suitable for classification with discrete features (e.g.
    word counts for text classification).
    """

    def __init__(self, alpha: float = 1.0) -> None:
        """Constructs a new Multinomial Naive Bayes classifier.

        Parameters
        ----------
        alpha : float, optional
            Laplace smoothing parameter, by default 1.0
        """
        super().__init__()
        self.alpha = alpha

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

        # Counts the classes and features
        self._feature_count = np.dot(y.T, x)
        self._class_count = np.sum(y, axis=0)

        # Applies Laplace smoothing
        smoothed_counts = self._feature_count + self.alpha
        total_count = np.sum(smoothed_counts, axis=1, keepdims=True)

        # Calculates log probabilities
        self._feature_log_prob = np.log(smoothed_counts) - np.log(total_count)
        self._class_log_prior = np.log(self._class_count) - np.log(np.sum(self._class_count))

    def predict_log_proba(self, x: ArrayLike) -> NDArray:   
        # Ensures inputs are Numpy arrays
        x = np.asarray(x)

        # Calculates the log likelihoods
        log_likelihoods = np.dot(x, self._feature_log_prob.T)
        log_posteriors = log_likelihoods + self._class_log_prior

        return log_posteriors
