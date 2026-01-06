"""Categorical Naive Bayes classifier class."""

import numpy as np
from numpy.typing import NDArray, ArrayLike
from .base import BaseNB

class CategoricalNB(BaseNB):
    """
    Naive Bayes classifier for categorical features.

    Suitable for classification with discrete features that are
    categorically distributed. The categories of each feature
    are drawn from a categorical distribution.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        """Constructs a new Categorical Naive Bayes classifier.

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
        n_classes = len(self._classes)

        # Calculates feature counts
        self.feature_count = []
        for i in range(self._n_features_in):
            # Extracts individual feature column
            x_feature = x[:, i].astype(int)

            # Counts each feature category for each class
            count_feature = []
            for j in range(n_classes):
                # Creates mask for the class
                mask = y[:, j].astype(bool)

                # Counts the number of each feature category for the class
                counts = np.bincount(x_feature[mask])
                count_feature.append(counts)
            
            # Ensures that all arrays are padded to the same length
            max_length = max(len(arr) for arr in count_feature)
            count_feature = np.vstack([
                np.pad(arr, (0, max_length - arr.shape[0])) for arr in count_feature
            ])
            self.feature_count.append(count_feature)

        self._class_count = np.sum(y, axis=0)

        # Calculates log probabilities for each feature
        self._feature_log_prob = []
        for i in range(self._n_features_in):
            # Applies Laplace smoothing to the feature count
            smoothed_count = self.feature_count[i] + self.alpha
            total_count = np.sum(smoothed_count, axis=1, keepdims=True)

            # Calculates the log probability for the feature
            log_probability = np.log(smoothed_count) - np.log(total_count)
            self._feature_log_prob.append(log_probability)
        
        # Calculates the log class prior
        self._class_log_prior = np.log(self._class_count) - np.log(np.sum(self._class_count))

    def predict_log_proba(self, x: ArrayLike) -> NDArray:   
        # Ensures inputs are Numpy arrays
        x = np.asarray(x)

        # Calculates the log likelihoods for each feature
        log_likelihoods = np.zeros((len(x), len(self._classes)))
        for i in range(self._n_features_in):
            category = x[:, i]
            log_likelihoods += self._feature_log_prob[i][:, category].T

        # Calculates the predictions for each test vector
        log_posteriors = log_likelihoods + self._class_log_prior

        return log_posteriors
