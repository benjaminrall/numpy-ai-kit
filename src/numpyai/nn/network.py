"""Main neural network class."""

from __future__ import annotations
import pickle
import numpy as np
from collections import defaultdict
from typing import Callable
from numpy.typing import NDArray
from numpyai.backend import ProgressBar, Representable
from .optimisers import Optimiser
from .losses import Loss
from .layers import Layer, TrainableLayer
from .metrics import Metric

class Network(Representable):
    """A neural network consisting of a linear stack of layers."""

    @property
    def built(self) -> bool:
        """Whether the network has been built."""
        return self._built
    
    @property
    def compiled(self) -> bool:
        """Whether the network has been compiled."""
        return self._compiled

    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers
        self.optimiser = None
        self.loss = None
        self.metrics = []
        self.input_shape = ()
        self.output_shape = ()
        self._built = False
        self._compiled = False

    def add(self, layer: Layer) -> None:
        """Adds a layer to the neural network.

        Parameters
        ----------
        layer : Layer
            A layer instance to be added
        """
        # Builds new layer if possible
        if self._built:
            self.output_shape = layer.build(self.output_shape)
        self.layers.append(layer)

    def pop(self) -> Layer | None:
        """Removes and returns the last layer in the network.

        Returns
        -------
        Layer | None
            The removed layer, or None if the network was empty.
        """
        if len(self.layers) == 0:
            return None
        layer = self.layers.pop()
        self.output_shape = layer.input_shape
        return layer
    
    def get_layer(self, index: int) -> Layer | None:
        """Retrieves a layer based on its index

        Parameters
        ----------
        index : int
            Index of the layer to retrieve.

        Returns
        -------
        Layer | None
            The layer instance at the given index,
            or None if the index was out of range.
        """
        if index < 0 or index >= len(self.layers):
            return None
        return self.layers[index]
    
    def build(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Builds the network based on the given input shape.

        Parameters
        ----------
        input_shape : tuple[int, ...]
            The shape of the input to the network.
            Should not include the batch dimension.

        Returns
        -------
        tuple[int, ...]
            The output shape of the network.
        """
        self.input_shape = input_shape
        self.output_shape = input_shape

        # Passes the input shape through the network, building all layers
        for layer in self.layers:
            self.output_shape = layer.build(self.output_shape)

        self._built = True
        return self.output_shape

    def compile(self, optimiser: str | Optimiser = 'rmsprop',
                loss: str | Loss = 'categorical_crossentropy',
                metrics: list[str | Metric | Callable[[NDArray, NDArray], float]] = []
                ) -> None:
        """Configures the network for training.

        Parameters
        ----------
        optimiser : str | Optimiser, optional
            Name of an optimiser or an optimiser instance, by default 'rmsprop'
        loss : str | Loss, optional
            Name of a loss function or a loss function instance, by default 'categorical_crossentropy'
        metrics : list, optional
            List of metric names, metric instances, or arbitrary metric functions to be used when
            evaluating the network, by default []
        """
        self.optimiser = Optimiser.get(optimiser)
        self.loss = Loss.get(loss)
        self.metrics = [Metric.get(metric) if isinstance(metric, str) else metric for metric in metrics]
        self._compiled = True

    def __call__(self, inputs: NDArray, **kwargs) -> NDArray:
        """Calls the network on a given set of inputs.

        Parameters
        ----------
        inputs : NDArray
            An array of inputs with the shape (batches, ...)

        Returns
        -------
        NDArray
            The outputs of the network.
        """
        return self.call(inputs, **kwargs)
    
    def call(self, inputs: NDArray, **kwargs) -> NDArray:
        """Calls the network on a given set of inputs.

        Parameters
        ----------
        inputs : NDArray
            An array of inputs with the shape (batches, ...)

        Returns
        -------
        NDArray
            The outputs of the network.
        """
        for layer in self.layers:
            inputs = layer(inputs, **kwargs)
        return inputs
    
    def fit(self, x: NDArray, y: NDArray, batch_size: int = 32, epochs: int = 1, verbose: bool = True,
            validation_split: float = 0, validation_data: tuple[NDArray, NDArray] | None = None, shuffle: bool = True) -> None:
        """Trains the network for a fixed number of epochs

        Parameters
        ----------
        x : NDArray
            Input data
        y : NDArray
            Target output data
        batch_size : int, optional
            Number of samples used per gradient update, by default 32
        epochs : int, optional
            Number of epochs to train the network, by default 1
        verbose : bool, optional
            Whether to train the network verbosely (with a progress bar), by default True
        validation_split : float, optional
            Fraction of the training data to be used as validation data, by default 0
        validation_data : NDArray | None, optional
            Validation data in the form (inputs, outputs), by default None
        shuffle : bool, optional
            Whether to shuffle the training data before each epoch, by default True
        """
        # Compiles the network with default settings if it hasn't been manually compiled
        if not self._compiled:
            self.compile()
        assert self.loss is not None, "Network must be compiled before use."
        assert self.optimiser is not None, "Network must be compiled before use."

        # Splits data into training and validation sets
        validation_x, validation_y = x, y
        if validation_data:
            validation_x, validation_y = validation_data
        elif 0 < validation_split < 1:
            validation_index = x.shape[0] - int(len(x) * validation_split)
            validation_x = x[validation_index:]
            validation_y = y[validation_index:]
            x = x[:validation_index]
            y = y[:validation_index]
        
        # Runs each training epoch
        for epoch in range(epochs):
            # Shuffles the training data before splitting it into batches
            if shuffle:
                p = np.random.permutation(x.shape[0])
                x, y = x[p], y[p]

            # Creates an iterator for the batch indices
            batch_indices = range(0, x.shape[0], batch_size)
            if verbose:
                batch_indices = ProgressBar(
                    batch_indices, 
                    desc="Epoch {:{}d}/{:d}".format(epoch + 1, len(str(epochs)), epochs),
                    bars=20, smoothing = 0, newline_close=False
                )

            # Performs the gradient updates for each batch using backpropagation
            for i in batch_indices:
                derivatives = self.loss.derivative(
                    self.call(x[i:i+batch_size], training=True),
                    y[i:i+batch_size]
                )

                for layer in reversed(self.layers):
                    derivatives = layer.backward(derivatives, self.optimiser)

            # Prints final loss and accuracy measurements for validation data
            if verbose:
                results = self.evaluate(validation_x, validation_y, batch_size, False)
                results_str = f" - " + " - ".join([f'loss: {results["loss"]:.8f}'] + [
                    f'{metric.__name__}: {results[metric.__name__]:.8f}' for metric in self.metrics
                ])
                print(results_str)


    def evaluate(self, x: NDArray, y: NDArray, batch_size: int = 32, verbose: bool = True) -> dict[str, float]:
        """Evaluates the performance of the network using its loss and all network metrics.

        Parameters
        ----------
        x : NDArray
            Input data
        y : NDArray
            Target output data
        batch_size : int, optional
            Number of samples used per computation batch, by default 32
        verbose : bool, optional
            Whether to evaluate the network verbosely (with a progress bar), by default True

        Returns
        -------
        dict[str, float]
            A dictionary mapping metric names to their evaluated values.
        """
        # Ensures network is compiled before evaluating it
        if not self._compiled:
            raise RuntimeError("Cannot evaluate a network that hasn't been compiled yet.")
        assert self.loss is not None, "Network must be compiled before use."
        assert self.optimiser is not None, "Network must be compiled before use."

        # Creates an iterator for the batch indices
        batch_indices = range(0, x.shape[0], batch_size)
        if verbose:
            batch_indices = ProgressBar(
                batch_indices, 
                desc="Evaluating",
                bars=20, smoothing = 0, newline_close=False
            )
        
        # Calculates the total loss and metric values over all batches
        totals: defaultdict[str, float] = defaultdict(float)
        for i in batch_indices:
            output = self.call(x[i:i+batch_size])
            target = y[i:i+batch_size]
            totals['loss'] += output.shape[0] * (self.loss(output, target) + self.penalty())
            for metric in self.metrics:
                totals[metric.__name__] += output.shape[0] * metric(output, target)

        # Calculates the mean value of the metrics over all given inputs
        for key in totals:
            totals[key] /= x.shape[0]

        if verbose:
            results_str = f" - " + " - ".join([f'loss: {totals["loss"]:.8f}'] + [
                f'{metric.__name__}: {totals[metric.__name__]:.8f}' for metric in self.metrics
            ])
            print(results_str)

        return totals
    
    def predict(self, x: NDArray, batch_size: int = 32, verbose: bool = True) -> NDArray:
        """Generates output predictions for a set of inputs.

        Parameters
        ----------
        x : NDArray
            Input data
        batch_size : int, optional
            Number of samples used per computation batch, by default 32
        verbose : bool, optional
            Whether to predict verbosely (with a progress bar), by default True

        Returns
        -------
        NDArray
            Array containing the network's predictions.
        """
        # Creates an iterator for the batch indices
        batch_indices = range(0, x.shape[0], batch_size)
        if verbose:
            batch_indices = ProgressBar(
                batch_indices, 
                desc="Predicting",
                bars=20, smoothing = 0, 
                newline_close=False
            )

        # Calculates output for all batches
        y = np.zeros(x.shape[:1] + self.layers[-1].output_shape)
        for i in batch_indices:
            y[i:i+batch_size] = self.call(x[i:i+batch_size])
        return y
    
    def penalty(self) -> float:
        """Calculates the regularisation penalty of all layers in the network.

        Returns
        -------
        float
            The network's total regularisation penalty.
        """
        return sum([layer.penalty() for layer in self.layers if isinstance(layer, TrainableLayer)])
    
    def summary(self) -> None:
        """Prints a string summary of the network."""
        # Ensures that the network is built before printing a summary
        if not self._built:
            raise ValueError(
                "This model has not yet been built. Build the model first "
                "by calling `build()` or by calling the model on a batch of data."
            )
        
        # Stores all information to be printed in the summary
        types = [type(layer).__name__ for layer in self.layers]
        shapes = [str(layer.output_shape) for layer in self.layers]
        params = [str(layer.parameters) if isinstance(layer, TrainableLayer) else '0' for layer in self.layers]

        # Calculates the sizes for each column in the summary table
        cols = [
            max(10, max([len(t) for t in types]) + 5),
            max(10, max([len(s) for s in shapes]) + 12),
            max(10, max([len(p) for p in params]) + 7)
        ]

        # Prints the column headers and divider bars
        print(f"\n{'':=^{sum(cols) + 8}s}")
        print(f" {'Layer':<{cols[0]}s} | {'Output Shape':^{cols[1]}s} | {'Param #':>{cols[2]}s} ")
        print(f"{'':=^{sum(cols) + 8}s}")

        # Prints layer information
        for t, s, p in zip(types, shapes, params):
            print(f" {t:<{cols[0]}s} | {s:^{cols[1]}s} | {p:>{cols[2]}s} ")
        
        # Prints total parameter value
        print(f"{'':=^{sum(cols) + 8}}")
        print(f"There are {sum([layer.parameters for layer in self.layers if isinstance(layer, TrainableLayer)])} total parameters.\n")

    def save(self, filepath: str) -> None:
        """Serialises and saves a network.

        Parameters
        ----------
        filepath : str
            Path to the location at which the network will be saved
        """
        # Ensures file is saved with the .npai extension
        if not filepath.endswith('.npai'):
            filepath += '.npai'

        # Saves the network as a binary file
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> Network:
        """Loads and deserialises a network.

        Parameters
        ----------
        filepath : str
            Path to the saved network

        Returns
        -------
        Network
            Network instance loaded from the file.
        """
        # Attempts to open the file and load the network
        try:
            with open(filepath, 'rb') as f:
                network = pickle.load(f)
        except OSError as e:
            raise OSError('Error loading network from file.') from e
        except pickle.UnpicklingError as e:
            raise OSError(
                'Error loading network from file. File specified is not a valid pickled binary file.'
            ) from e
        return network
    
    def get_variables(self) -> list[list[NDArray]]:
        """Retrieves the variables of all layers in the network.

        Returns
        -------
        list[list[NDArray]]
            A list containing the variables of each layer.
        """
        return [layer.get_variables() for layer in self.layers if isinstance(layer, TrainableLayer)]
    
    def set_variables(self, variables: list[list[NDArray]]) -> None:
        """Sets the variables of the network, from a list of layer variables.

        Parameters
        ----------
        variables : list[list[NDArray]]
            A list containing the variables of each layer
        """
        trainable_layers = [layer for layer in self.layers if isinstance(layer, TrainableLayer)]
        if len(variables) != len(trainable_layers):
            raise ValueError('Variables list must contain variables for each trainable layer in the network.')
        for layer, layer_vars in zip(trainable_layers, variables):
            layer.set_variables(layer_vars)

    def save_variables(self, filepath: str) -> None:
        """Serialises and saves a network's variables.

        Parameters
        ----------
        filepath : str
            Path to the location at which the variables will be saved
        """
        # Ensures file is saved with the .npaiw file extension
        if not filepath.endswith('.npaiw'):
            filepath += '.npaiw'

        # Saves the variables as a binary file
        with open(filepath, 'wb') as f:
            pickle.dump(self.get_variables(), f)

    def load_variables(self, filepath: str) -> None:
        """Loads and deserialises a network's variables.

        Parameters
        ----------
        filepath : str
            Path to the saved variables
        """
        # Attempts to open the file and load the variables
        try:
            with open(filepath, 'rb') as f:
                variables = pickle.load(f)
        except OSError as e:
            raise OSError('Error loading variables from file.') from e
        except pickle.UnpicklingError as e:
            raise OSError(
                'Error loading variables from file. File specified is not a valid pickled binary file.'
            ) from e
        
        # Sets variables for the network
        self.set_variables(variables)
