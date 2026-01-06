import numpy as np
from numpy.typing import NDArray
from pathlib import Path

EPSILON = 1e-12
"""A small constant used for numerical stability."""

CACHE_DIR = Path.home() / '.numpyai'
"""Location in the user's home directory used for caching."""

def clip_epsilon(x: NDArray) -> NDArray:
    """Element-wise value clipping of a Numpy array to the range [epsilon, 1 - epsilon]."""
    return np.clip(x, EPSILON, 1 - EPSILON)

def normalise_subarrays(x: NDArray) -> NDArray:
    """Normalises subarrays along the last dimension of the input so they each sum to 1."""
    return x / (np.sum(x, axis=-1, keepdims=True) + EPSILON)

def one_hot_encode(x: NDArray, classes: int = -1) -> NDArray:
    """Performs one-hot encoding on an input array with a given number of classes.

    Parameters
    ----------
    x : NDArray
        Input to be encoded.
    classes : int, optional
        The number of classes. If `classes` is -1 or is invalid, then
        it is automatically recalculated as `max(x) + 1`.

    Returns
    -------
    NDArray
        The result of perfoming one-hot encoding on the input.
    """
    classes = max(np.max(x) + 1, classes)
    one_hot = np.zeros((x.size, classes))
    one_hot[np.arange(x.size), x] = 1
    return one_hot

def dilate_2d(x: NDArray, dilation_factor: tuple[int, int]) -> NDArray:
    """Dilates a 2D array by inserting gaps between its rows and columns.

    Parameters
    ----------
    x : NDArray
        The input array with shape (batches, rows, cols, ...).
    dilation_factor : tuple[int, int]
        The factors by which to dilate the array in each direction.

    Returns
    -------
    NDArray
        The dilated array.
    """
    # Case in which no dilation takes place
    if dilation_factor == (1, 1):
        return x

    dy, dx = dilation_factor
    if dy <= 0 or dx <= 0:
        raise ValueError(f"Dilation factors must be positive integers. Got: {dilation_factor}")
    
    # Calculates resulting shape of the dilated tensor
    batches, rows, cols = x.shape[:3]
    dilated_rows = rows + (rows - 1) * (dy - 1)
    dilated_cols = cols + (cols - 1) * (dx - 1)
    dilated_shape = (batches, dilated_rows, dilated_cols)

    # Constructs the dilated tensor
    dilated_tensor = np.zeros(dilated_shape + x.shape[3:])
    dilated_tensor[:, ::dy, ::dx] = x
    return dilated_tensor