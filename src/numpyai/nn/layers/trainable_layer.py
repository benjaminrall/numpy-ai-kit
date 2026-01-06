"""Base trainable layer class."""

from numpy.typing import NDArray
from .layer import Layer

class TrainableLayer(Layer):
    """Abstract base class from which all layers with trainable variables inherit."""

    @property
    def trainable(self) -> bool:
        return True

    @property
    def n_variables(self) -> int:
        """Number of variables that the layer contains."""
        return len(self._variables)

    def __init__(self) -> None:
        super().__init__()
        self.parameters: int = 0
        self._variables: list[NDArray] = []

    def penalty(self) -> float:
        """Calculates the regularisation penalty of the layer's variables."""
        return 0

    def get_variables(self) -> list[NDArray]:
        """Retrieves the variables of the layer as a list of Numpy arrays."""
        if not self._built:
            raise RuntimeError("Cannot retrieve variables from a layer that hasn't been built yet.")
        return self._variables.copy()
    
    def set_variables(self, variables: list[NDArray]) -> None:
        """Sets the variables of the layer from a list of Numpy arrays."""
        if not self._built:
            raise RuntimeError("Cannot set variables for a layer that hasn't been built yet.")
        if len(variables) != self.n_variables or not all([a.shape == b.shape for a, b in zip(self._variables, variables)]):
            raise ValueError("Provided variables do not match the layer's specification.")
        self._variables = variables