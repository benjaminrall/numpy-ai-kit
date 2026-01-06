"""Complement Naive Bayes classifier class."""

import numpy as np
from numpy.typing import NDArray, ArrayLike
from .base import BaseNB

class ComplementNB(BaseNB):
    """
    Complement Naive Bayes classifier for multinomial models.

    Designed to correct assumptions made by the standard
    Multinomial Naive Bayes classifier, and is particularly
    suited for imbalanced data sets.
    """

    def __init__(self, alpha: float = 1.0, norm: bool = False) -> None:
        """Constructs a new Complement Naive Bayes classifier.

        Parameters
        ----------
        alpha : float, optional
            Laplace smoothing parameter, by default 1.0
        norm : bool, optional
            Whether to perform a second normalisation of the weights.
        """
        super().__init__()
        self.alpha = alpha
        self.norm = norm

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
        self._feature_all = np.sum(self._feature_count, axis=0)
        self._class_count = np.sum(y, axis=0)

        # Calculates complement count and applies Laplace smoothing
        comp_count = self._feature_all + self.alpha - self._feature_count
        total_count = np.sum(comp_count, axis=1, keepdims=True)

        # Calculates log probabilities
        feature_log_prob = np.log(comp_count) - np.log(total_count)
        self._class_log_prior = np.log(self._class_count) - np.log(np.sum(self._class_count))

        # Normalises feature log probabilities
        if self.norm:
            summed_feature_prob = np.sum(feature_log_prob, axis=1, keepdims=True)
            self._feature_log_prob = feature_log_prob / summed_feature_prob
        else:
            self._feature_log_prob = -feature_log_prob


    def predict_log_proba(self, x: ArrayLike) -> NDArray:   
        # Ensures inputs are Numpy arrays
        x = np.asarray(x)

        # Calculates the log likelihoods
        log_likelihoods = np.dot(x, self._feature_log_prob.T)
        if len(self._classes) == 1:
            log_likelihoods += self._class_log_prior
        
        return log_likelihoods
