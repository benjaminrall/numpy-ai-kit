"""Ordinal encoder class."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray, ArrayLike
from numpyai.backend import Representable

class OrdinalEncoder(Representable):
    """Encodes categorical features as an integer array."""

    def __init__(self, encoded_missing_value: float = np.nan) -> None:
        """Constructs a new OrdinalEncoder instance.

        Parameters
        ----------
        encoded_missing_value : float, optional
            Encoded value of missing categories, by default np.nan
        """
        self.encoded_missing_value = encoded_missing_value
        self._mapping: list[dict] = []
        self._categories: list[NDArray] = []

    def fit(self, x: ArrayLike) -> OrdinalEncoder:
        """Fits the ordinal encoder to a given array of categorical feature vectors.

        Parameters
        ----------
        x : ArrayLike
            Array of shape (n_samples, n_features) used to determine 
            the categories 

        Returns
        -------
        OrdinalEncoder
            Fitted ordinal encoder instance.
        """
        # Ensures input is a Numpy array
        x = np.asarray(x, dtype=object)

        self._n_features_in = x.shape[1]
        self._mapping = []
        self._categories = []

        for fi in range(self._n_features_in):
            # Extracts all possible feature elements from the input vectors
            feature = x[:, fi]
            categories = sorted(set(feature))

            # Stores mapping and categories for the feature
            self._mapping.append({f: i for i, f in enumerate(categories)})
            self._categories.append (np.array(categories))

        return self
    
    def transform(self, x: ArrayLike) -> NDArray:
        """Transforms the given categorical feature vectors into an integer array.

        Parameters
        ----------
        x : NDArray
            Array of shape (n_samples, n_features) containing the categorical
            feature vectors to be encoded

        Returns
        -------
        NDArray
            Array of shape (n_samples, n_features) containing the result
            of encoding the input vectors.
        """
        # Ensures input is a Numpy array
        x = np.asarray(x, dtype=object)

        # Creates array to hold encoded data
        encoded_data = np.empty((len(x), self._n_features_in), dtype=int)
        
        # Encodes each feature in the input vectors array
        for i in range(self._n_features_in):
            # Gets the mapping for the specific feature
            feature_mapping = self._mapping[i]

            # Encodes the feature column
            encode = np.vectorize(feature_mapping.get)
            encoded_data[:, i] = encode(x[:, i], self.encoded_missing_value)

        return encoded_data
    
    def fit_transform(self, x: ArrayLike) -> NDArray:
        """Fits the encoder and transforms the given data.

        Parameters
        ----------
        x : ArrayLike
            Array of shape (n_samples, n_features) containing the categorical
            feature vectors to be encoded

        Returns
        -------
        NDArray
            Array of shape (n_samples, n_features) containing the result
            of encoding the input vectors.
        """
        return self.fit(x).transform(x)
    
    def inverse_transform(self, x: ArrayLike) -> NDArray:
        """Transforms encoded data back into its original representation.

        Parameters
        ----------
        x : ArrayLike
            Array of shape (n_samples, n_features) containing the encoded
            data to be reverted to its original form

        Returns
        -------
        NDArray
            Array of shape (n_samples, n_features) containing the 
            decoded categorical feature vectors.
        """
        # Ensures input is a Numpy array
        x = np.asarray(x)

        # Creates array to hold data data
        decoded_data = np.empty((len(x), self._n_features_in), dtype=object)
        
        # Decodes each feature in the input vectors array
        for i in range(self._n_features_in):
            # Gets the categories for the specific feature
            feature_categories = self._categories[i]

            # Decodes the feature column
            decode = np.vectorize(feature_categories.__getitem__)
            decoded_data[:, i] = decode(x[:, i])

        return decoded_data
