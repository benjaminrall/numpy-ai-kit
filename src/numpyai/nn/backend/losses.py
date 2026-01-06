"""Neural network loss functions."""

import numpy as np
from numpy.typing import NDArray
from numpyai.backend import clip_epsilon, normalise_subarrays
from .activations import softmax

def normalise_output(output: NDArray, from_logits: bool) -> NDArray:
    """Normalises a given output array to be used in a loss function."""
    return softmax(output) if from_logits else normalise_subarrays(output)

def binary_crossentropy(output: NDArray, target: NDArray, from_logits: bool = False) -> float:
    """Calculates the binary cross-entropy loss between an output and target."""
    # Use separate stable calculation when using logits
    if from_logits:
        log_result = np.log(1 + np.exp(-np.abs(output)))
        return float(np.mean(np.maximum(output, 0) - output * target + log_result))

    # Calculates binary cross-entropy from probability distribution
    output = clip_epsilon(output)
    return float(-np.mean(target * np.log(output) + (1 - target) * np.log(1 - output)))

def categorical_crossentropy(output: NDArray, target: NDArray, from_logits: bool = False) -> float:
    """Calculates the categorical cross-entropy loss between an output and target."""
    output = clip_epsilon(normalise_output(output, from_logits))
    return -np.mean(np.sum(target * np.log(output), axis=-1))

def mean_absolute_error(output: NDArray, target: NDArray) -> float:
    """Calculates the mean absolute error loss between an output and target."""
    return float(np.mean(np.abs(output - target)))

def mean_squared_error(output: NDArray, target: NDArray) -> float:
    """Calculates the mean squared error loss between an output and target."""
    return float(np.mean(np.square(output - target)))
