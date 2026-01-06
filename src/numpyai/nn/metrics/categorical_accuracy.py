"""Categorical accuracy metric class."""

import numpy as np
from numpy.typing import NDArray
from .metric import Metric

class CategoricalAccuracy(Metric):
    """
    Calculates how often outputs match one-hot labels.    

    Works with both logits and probabilities as predicted outputs.
    """

    identifier = 'categorical_accuracy'

    def call(self, output: NDArray, target: NDArray) -> float:
        """Calculates the categorical accuracy."""
        return np.mean(np.argmax(output, axis=-1) == np.argmax(target, axis=-1))
