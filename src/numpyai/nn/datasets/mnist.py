"""MNIST handwritten digits dataset."""

import numpy as np
from numpy.typing import NDArray
from .dataset import Dataset
from .utils import get_file

class MNIST(Dataset):
    """MNIST handwritten digits dataset."""

    identifier = 'mnist'

    @staticmethod
    def load_data(path: str = 'mnist.npz') -> tuple[tuple[NDArray, NDArray], tuple[NDArray, NDArray]]:
        """Loads the MNIST dataset.

        This is a dataset of 60000 28x28 grayscale images of the 10 digits,
        along with a test set of 10000 images.
        More info can be found at the 
        [MNIST homepage](http://yann.lecun.com/exdb/mnist/).
        
        Parameters
        ----------
        path : str, optional
            Path where to cache the dataset locally 
            (relative to ~/.pyai/datasets), by default 'mnist.npz'
        
        Returns
        -------
        tuple[tuple[NDArray, NDArray], tuple[NDArray, NDArray]]
            Tuple of NumPy arrays: `(x_train, y_train), (x_test, y_test)`.
        
            **`x_train`**: `uint8` NumPy array of grayscale image data with shapes
            `(60000, 28, 28)`, containing the training data. Pixel values range
            from 0 to 255.
            
            **`y_train`**: `uint8` NumPy array of digit labels (integers in range 0-9)
            with shape `(60000,)` for the training data.

            **`x_test`**: `uint8` NumPy array of grayscale image data with shapes
            `(10000, 28, 28)`, containing the test data. Pixel values range
            from 0 to 255.

            **`y_test`**: `uint8` NumPy array of digit labels (integers in range 0-9)
            with shape `(10000,)` for the test data.

        License
        -------
        Yann LeCun and Corinna Cortes hold the copyright of MNIST dataset,
        which is a derivative work from original NIST datasets.
        MNIST dataset is made available under the terms of the
        [Creative Commons Attribution-Share Alike 3.0 license](
            https://creativecommons.org/licenses/by-sa/3.0/).
        Description of dataset taken from the [Keras API reference](
        https://keras.io/api/).
        """
        origin_folder = (
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
        )

        file = get_file(
            fname=path,
            origin=origin_folder + 'mnist.npz',
            file_hash=(
                "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
            )
        )

        with np.load(file, allow_pickle=True) as f:
            x_train, y_train = f["x_train"], f["y_train"]
            x_test, y_test = f["x_test"], f["y_test"]
            return (x_train, y_train), (x_test, y_test)