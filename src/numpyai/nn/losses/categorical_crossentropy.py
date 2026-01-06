"""Categorical cross-entropy loss function class."""

from numpy.typing import NDArray
from numpyai.nn.backend import categorical_crossentropy, normalise_output
from .loss import Loss

class CategoricalCrossentropy(Loss):
    """Computes the cross-entropy loss between the outputs and targets."""

    identifier = 'categorical_crossentropy'
    aliases = ['cce']

    def __init__(self, from_logits: bool = False) -> None:
        self.from_logits = from_logits

    def call(self, output: NDArray, target: NDArray) -> float:
        """Calculates the result of the categorical cross-entropy loss function."""
        return categorical_crossentropy(output, target, self.from_logits)

    def derivative(self, output: NDArray, target: NDArray) -> NDArray:
        """Calculates the derivative of the categorical cross-entropy loss function."""
        return normalise_output(output, self.from_logits) - target
