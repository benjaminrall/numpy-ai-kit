"""Binary cross-entropy metric class."""

from numpy.typing import NDArray
from typing import Optional
from numpyai.nn.backend import binary_crossentropy
from .metric import Metric

class BinaryCrossentropy(Metric):
    """Computes the binary cross-entropy between the outputs and targets."""

    identifier = 'binary_crossentropy'
    aliases = ['bce']

    def __init__(self, display_name: Optional[str] = None, from_logits: bool = False) -> None:
        super().__init__(display_name)
        self.from_logits = from_logits

    def call(self, output: NDArray, target: NDArray) -> float:
        """Calculates the binary cross-entropy."""
        return binary_crossentropy(output, target, self.from_logits)