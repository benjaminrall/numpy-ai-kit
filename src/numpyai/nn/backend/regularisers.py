"""Neural network regularisation functions."""

import numpy as np
from numpy.typing import NDArray

def l1(factor: float, x: NDArray) -> float:
    """Calculates the L1 regularisation penalty of a given input."""
    return factor * np.sum(np.abs(x))

def l2(factor: float, x: NDArray) -> float:
    """Calculates the L2 regularisation penalty of a given input."""
    return factor * np.sum(np.square(x))
