"""Neural network activation functions."""

import numpy as np
from numpy.typing import NDArray

def leaky_relu(x: NDArray, alpha: float = 0.3) -> NDArray:
    """Applies the leaky version of the rectified linear unit activation function to the input array."""
    return np.maximum(x, alpha * x)

def relu(x: NDArray) -> NDArray:
    """Applies the rectified linear unit activation function to the input array."""
    return np.maximum(x, 0)

def sigmoid(x: NDArray) -> NDArray:
    """Applies the sigmoid activation function to the input array."""
    return 1.0 / (1.0 + np.exp(-x))

def stable_sigmoid(x: NDArray) -> NDArray:
    """Applies a numerically stable sigmoid activation function to the input array."""
    # Calculates exponentials for negative and positive values separately
    neg_exp = np.exp(x[x < 0])
    pos_exp = np.exp(-x[x >= 0])

    # Creates output array using separate equations for stability
    z = np.zeros(x.shape)
    z[x < 0] = neg_exp / (1.0 + neg_exp)
    z[x >= 0] = 1.0 / (1.0 + pos_exp)
    return z

def softmax(x: NDArray) -> NDArray:
    """Applies the softmax activation function to the input array."""
    # Ensure numerical stability by subtracting the maximum value from the input
    max_val = np.max(x, axis=-1, keepdims=True)

    # Calculate exponentials and their sums
    exps = np.exp(x - max_val)
    sums = np.sum(exps, axis=-1, keepdims=True)

    # Return final probabilities array
    return exps / sums
