"""Count vectoriser class."""

from __future__ import annotations
import re
import numpy as np
from numpy.typing import NDArray

class CountVectoriser:
    """Converts a list of strings to a matrix of token counts."""

    def __init__(self, lowercase: bool = True, binary: bool = False) -> None:
        """Creates a CountVectoriser object.

        Parameters
        ----------
        lowercase : bool
            Whether to convert all characters to lowercase before tokenising
        binary : bool
            If True, all non zero counts are set to 1. This is useful for discrete probabilistic models
            that model binary events rather than integer counts.
        """
        self.lowercase = lowercase
        self.binary = binary

        # Stores the default regular expression to be used as a tokeniser
        self.tokeniser = re.compile(r"(?u)\b\w\w+\b").findall

        self.vocabulary = {}

    def fit(self, raw_documents: list[str]) -> CountVectoriser:
        """Creates a vocabulary dictionary of all tokens in the documents.

        Parameters
        ----------
        raw_documents : list[str]
            List of strings used to create the token vocabulary for the vectoriser.

        Returns
        -------
        CountVectoriser
            Fitted count vectoriser instance.
        """
        # Creates a set containing the unique words in the documents
        words = set()
        for document in raw_documents:
            # Convert the document to lowercase if specified
            if self.lowercase:
                document = document.lower()

            # Adds the tokens to the words set
            for token in self.tokeniser(document):
                words.add(token)

        # Constructs a vocabulary where each word is mapped to its corresponding index
        self.vocabulary = {word: i for i, word in enumerate(sorted(words))}

        return self

    def transform(self, raw_documents: list[str]) -> NDArray:
        """Transforms documents to a document-term matrix.

        Parameters
        ----------
        raw_documents : list[str]
            List of strings to convert to feature vectors.

        Returns
        -------
        NDArray
            Document-term matrix containing feature vectors for each
            input document.
        """
        # Creates count vectors array
        count_vectors = np.zeros((len(raw_documents), len(self.vocabulary)), dtype=int)

        # Counts tokens in each document
        for i, document in enumerate(raw_documents):
            # Convert the document to lowercase if specified
            if self.lowercase:
                document = document.lower()

            # Increment the count for each token that is in the vocabulary
            for token in self.tokeniser(document):
                if token in self.vocabulary:
                    count_vectors[i, self.vocabulary[token]] += 1

        # Makes the count vectors binary if specified
        if self.binary:
            count_vectors = (count_vectors > 0).astype(int)

        return count_vectors

    def fit_transform(self, raw_documents: list[str]) -> NDArray:
        """Creates a vocabulary dictionary and returns document-term matrix.

        Parameters
        ----------
        raw_documents : list[str]
            List of strings to convert to feature vectors.

        Returns
        -------
        NDArray
            Document-term matrix containing feature vectors for each
            input document.
        """
        return self.fit(raw_documents).transform(raw_documents)

    def inverse_transform(self, x: NDArray) -> list[NDArray]:
        """Returns the tokens represented by each of the given feature vectors.

        Parameters
        ----------
        x : NDArray
            Feature vectors with the shape (n_samples, n_features)

        Returns
        -------
        list[NDArray]
            List of arrays containing the tokens that are present in each
            input feature vector.
        """
        words = np.array(list(self.vocabulary.keys()))
        return [words[sample >= 1] for sample in x]

    def get_feature_names(self) -> NDArray:
        """Returns output feature names for transformation."""
        return np.array([key for key in self.vocabulary.keys()])

    def get_vocabulary(self) -> dict:
        """Returns the vectoriser's vocabulary dictionary."""
        return self.vocabulary
