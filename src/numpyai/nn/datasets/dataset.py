"""Base dataset class."""

from abc import ABC, abstractmethod
from numpy.typing import NDArray
from typing import Optional
from numpyai.backend import Registrable

class Dataset(Registrable['Dataset'], ABC):
    """Abstract base class from which all neural network metrics inherit."""

    identifier: str
    """The dataset's identifier."""

    @staticmethod
    def load(dataset: str, path: Optional[str] = None) -> tuple[tuple[NDArray, NDArray], tuple[NDArray, NDArray]]:
        """Loads data from the given dataset identifier, optionally passing through a path."""
        d = Dataset.get(dataset)
        if path is None:
            return d.load_data()
        else:
            return d.load_data(path)

    @staticmethod
    @abstractmethod
    def load_data(path: str = '') -> tuple[tuple[NDArray, NDArray], tuple[NDArray, NDArray]]:
        """Loads the dataset."""