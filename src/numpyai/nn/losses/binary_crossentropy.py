"""Binary cross-entropy loss function class."""

from numpy.typing import NDArray
from numpyai.nn.backend import binary_crossentropy
from .loss import Loss

class BinaryCrossentropy(Loss):
    """
    Computes the cross-entropy loss between the outputs and targets.

    This cross-entropy loss is used for binary classification applications
    where the target outputs are either 0 or 1.
    """

    identifier = 'binary_crossentropy'
    aliases = ['bce']

    def __init__(self, from_logits: bool = False) -> None:
        self.from_logits = from_logits

    def call(self, output: NDArray, target: NDArray) -> float:
        """Calculates the result of the binary cross-entropy loss function."""
        return binary_crossentropy(output, target, self.from_logits)

    def derivative(self, output: NDArray, target: NDArray) -> NDArray:
        """Calculates the derivative of the binary cross-entropy loss function."""
        return output - target
