"""Categorical cross-entropy metric class."""

from numpy.typing import NDArray
from typing import Optional
from numpyai.nn.backend import categorical_crossentropy
from .metric import Metric

class CategoricalCrossentropy(Metric):
    """Computes the categorical cross-entropy between the outputs and targets."""

    identifier = 'categorical_crossentropy'
    aliases = ['cce']

    def __init__(self, display_name: Optional[str] = None, from_logits: bool = False) -> None:
        super().__init__(display_name)
        self.from_logits = from_logits

    def call(self, output: NDArray, target: NDArray) -> float:
        """Calculates the categorical cross-entropy."""
        return categorical_crossentropy(output, target, self.from_logits)
