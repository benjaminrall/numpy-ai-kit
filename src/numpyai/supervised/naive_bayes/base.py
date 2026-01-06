"""Base Naive Bayes classifier class."""

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray, ArrayLike
from numpyai.backend import Representable
from numpyai.preprocessing import LabelBinariser

class BaseNB(Representable, ABC):
    """Abstract base class from which all Naive Bayes classifiers inherit."""

    def __init__(self) -> None:
        self._classes = np.empty(0)
        self._binariser = LabelBinariser()

    @abstractmethod
    def fit(self, x: ArrayLike, y: ArrayLike) -> None:
        """Fits the classifier according to the given training data and labels.

        Parameters
        ----------
        x : ArrayLike
            Training vectors with the shape (n_samples, n_features)
        y : ArrayLike
            Target labels with the shape (n_samples,)
        """

    @abstractmethod
    def predict_log_proba(self, x: ArrayLike) -> NDArray:
        """Generates output log probabilites for a set of inputs.

        Parameters
        ----------
        x : ArrayLike
            Test vectors with the shape (n_samples, n_features)

        Returns
        -------
        NDArray
            Predicted probability values for `x`.
        """

    def predict_proba(self, x: ArrayLike) -> NDArray:
        """Generates output probabilities for a set of inputs.
        
        Parameters
        ----------
        x : ArrayLike
            Test vectors with the shape (n_samples, n_features)

        Returns
        -------
        NDArray
            Predicted probability values for `x`.
        """
        # Calculates the log posterior probabilities
        log_posteriors = self.predict_log_proba(x)
        
        # Converts log posteriors to output probabilities using softmax
        exp_posteriors = np.exp(log_posteriors - np.max(log_posteriors, axis=1, keepdims=True))
        probabilities = exp_posteriors / np.sum(exp_posteriors, axis=1, keepdims=True)

        return probabilities
        
    def predict(self, x: ArrayLike) -> NDArray:
        """Generates output predictions for a set of inputs.

        Parameters
        ----------
        x : ArrayLike
            Test vectors with the shape (n_samples, n_features)

        Returns
        -------
        NDArray
            Predicted target values for `x`.
        """
        log_posteriors = self.predict_log_proba(x)
        predictions = np.argmax(log_posteriors, axis=1)
        return self._classes[predictions]

    def evaluate(self, x: NDArray, y: NDArray) -> float:
        """Returns the accuracy for the given test data and labels.

        Parameters
        ----------
        x : NDArray
            Test vectors with the shape (n_samples, n_features) 
        y : NDArray
            True labels for `x` with the shape (n_samples,)

        Returns
        -------
        float
            The mean accuracy of the classifier over all input samples.
        """
        return np.mean(self.predict(x) == y)