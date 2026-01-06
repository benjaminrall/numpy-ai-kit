"""Bernoulli Naive Bayes classifier class."""

import numpy as np
from numpy.typing import NDArray, ArrayLike
from .base import BaseNB

class BernoulliNB(BaseNB):
    """
    Naive Bayes classifier for multivariate Bernoulli models.
    
    Designed for discrete data with binary/boolean features; if given
    any other type of data then it may binarise its input (depending on
    the `binarise_threshold` parameter).
    """

    def __init__(self, alpha: float = 1.0, binarise_threshold: float = 0.0) -> None:
        """Constructs a new Bernoulli Naive Bayes classifier.

        Parameters
        ----------
        alpha : float, optional
            Laplace smoothing parameter, by default 1.0
        binarise_threshold : float, optional
            Threshold for binarising sample featuers, by default 0.0
        """
        super().__init__()
        self.alpha = alpha
        self.binarise_threshold = binarise_threshold

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

        # Stores number of input features
        self._n_features_in = x.shape[1]

        # Binarises input vectors
        x = np.astype(x > self.binarise_threshold, int, copy=False)

        # Counts classes and features
        self._feature_count = np.dot(y.T, x)
        self._class_count = np.sum(y, axis=0)

        # Applies Laplace smoothing to both counts
        smoothed_fc = self._feature_count + self.alpha
        smoothed_cc = self._class_count + self.alpha * 2

        # Calculates log probabilities
        self._feature_log_prob = np.log(smoothed_fc) - np.log(np.reshape(smoothed_cc, (-1, 1)))
        self._class_log_prior = np.log(self._class_count) - np.log(np.sum(self._class_count))
        
    def predict_log_proba(self, x: ArrayLike) -> NDArray:
        # Ensures inputs are Numpy arrays
        x = np.asarray(x)

        # Binarises input vectors
        x = np.astype(x > self.binarise_threshold, int, copy=False)

        # Computes log likelihoods using negated probability
        neg_prob = np.log(1 - np.exp(self._feature_log_prob))
        log_likelihoods = np.dot(x, (self._feature_log_prob - neg_prob).T)
        log_posteriors = log_likelihoods + self._class_log_prior + np.sum(neg_prob, axis=1)

        return log_posteriors
