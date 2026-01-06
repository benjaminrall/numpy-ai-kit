"""Label binariser method and class."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray, ArrayLike
from numpyai.backend import Representable

def label_binarise(y: ArrayLike, classes: ArrayLike) -> NDArray:
    """Binarises labels in a one-vs-all fashion.

    Parameters
    ----------
    y : ArrayLike
        Array of shape (n_samples,) containing classes to be binarised
    classes : ArrayLike | None, optional
        Array of shape (n_classes,) holding a unique label for each class

    Returns
    -------
    NDArray
        Array of shape (n_samples, n_classes) containing the
        result of converting the input classes to binary labels.
        Shape will be (n_samples, 1) for binary problems.
    """
    # Ensures inputs are Numpy arrays
    y = np.asarray(y)
    classes = np.asarray(classes)

    # Binary classes are transformed to a column vector
    if len(classes) == 2:
        binary_labels = np.astype(y == classes[1], int, copy=False)
        return binary_labels.reshape((-1, 1))
    
    # Generates binary labels for generic multi-class case
    binary_labels = np.array([y == c for c in classes], dtype=int).T
    return binary_labels

class LabelBinariser(Representable):
    """Binarises labels in a one-vs-all fashion."""

    def __init__(self) -> None:
        self._classes = np.empty(0)

    def fit(self, y: ArrayLike) -> LabelBinariser:
        """Fits the label binariser to a given array of classes.

        Parameters
        ----------
        y : ArrayLike
            Array of shape (n_samples,) containing the classes
            used to fit the label binariser

        Returns
        -------
        LabelBinariser
            Fitted label binariser instance.
        """
        # Creates a sorted list of the unique classes in the input
        self._classes = np.array(sorted(set(np.asarray(y))))
        return self
    
    def transform(self, y: ArrayLike) -> NDArray:
        """Transforms multi-class labels to binary labels.

        Parameters
        ----------
        y : ArrayLike
            Array of shape (n_samples,) containing classes to be
            transformed to binary labels

        Returns
        -------
        NDArray
            Array of shape (n_samples, n_classes) containing the
            result of converting the input classes to binary labels.
            Shape will be (n_samples, 1) for binary problems.
        """
        # Ensures `y` is a Numpy array
        y = np.asarray(y)

        # Binary classes are transformed to a column vector
        if len(self._classes) == 2:
            binary_labels = np.astype(y == self._classes[1], int, copy=False)
            return binary_labels.reshape((-1, 1))
        
        # Generates binary labels for generic multi-class case
        binary_labels = np.array([y == c for c in self._classes], dtype=int).T
        return binary_labels

    def fit_transform(self, y: ArrayLike) -> NDArray:
        """Fits the binariser to and transforms a given list of classes.

        Parameters
        ----------
        y : ArrayLike
            Array of shape (n_samples,) containing classes to be
            transformed to binary labels

        Returns
        -------
        NDArray
            Array of shape (n_samples, n_classes) containing the
            result of converting the input classes to binary labels.
            Shape will be (n_samples, 1) for binary problems.
        """
        return self.fit(y).transform(y)
    
    def inverse_transform(self, y: ArrayLike, threshold: float = 0.5) -> NDArray:
        """Transforms binary labels back to multi-class labels.

        Parameters
        ----------
        y : ArrayLike
            Binary labels to be converted back to original form,
            with shape (n_samples, n_classes)
        threshold : float
            Threshold used to convert labels to binary. 

        Returns
        -------
        NDArray
            Array with shape (n_samples,) containing the class
            represented by each binary label.
        """
        # Ensures `y` is a binary Numpy array
        y = np.astype(np.asarray(y) > threshold, int, copy=False)

        # Inverse in case of binary classes
        if len(self._classes) == 2:
            return self._classes[y].reshape(-1)
        
        # Inverse of binary labels for generic multi-class case
        return np.array([self._classes[np.where(row == 1)][0] for row in y])
