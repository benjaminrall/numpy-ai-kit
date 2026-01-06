"""Binary accuracy metric class."""

import numpy as np
from numpy.typing import NDArray
from typing import Optional
from .metric import Metric

class BinaryAccuracy(Metric):
    """
    Calculates how often outputs match binary labels.
    
    Uses a threshold (default 0.5) to decide whether prediction values are 1 or 0.
    """

    identifier = 'binary_accuracy'

    def __init__(self, display_name: Optional[str] = None, threshold: float = 0.5) -> None:
        super().__init__(display_name)
        self.threshold = threshold

    def call(self, output: NDArray, target: NDArray) -> float:
        """Calculates the binary accuracy."""
        return np.mean((output >= self.threshold) == target)