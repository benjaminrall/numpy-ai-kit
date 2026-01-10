"""Binary accuracy metric class."""

import numpy as np
from numpy.typing import NDArray
from typing import Optional
from .metric import Metric

class BinaryAccuracy(Metric):
    """Calculates how often outputs match binary labels."""

    identifier = 'binary_accuracy'

    def __init__(self, display_name: Optional[str] = None, threshold: float = 0.5) -> None:
        """Calculates how often outputs match binary labels.

        Parameters
        ----------
        threshold : float, optional
            Threshold to decide whether prediction values are treated as 1 or 0.
            Defaults to 0.5.
        """
        super().__init__(display_name)
        self.threshold = threshold

    def call(self, output: NDArray, target: NDArray) -> float:
        """Calculates the binary accuracy."""
        return np.mean((output >= self.threshold) == target)