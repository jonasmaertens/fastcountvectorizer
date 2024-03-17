import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from ._count_vocab_cy import _count_vocab_cy


def _document_frequency(X):
    """Count the number of non-zero values for each feature in sparse X."""
    if sp.isspmatrix_csr(X):
        return np.bincount(X.indices, minlength=X.shape[1])
    else:
        return np.diff(X.indptr)


def _check_iterable(raw_documents):
    if isinstance(raw_documents, pd.Series):
        return raw_documents.to_numpy()
    elif isinstance(raw_documents, np.ndarray):
        return raw_documents
    elif isinstance(raw_documents, list):
        return np.array(raw_documents)
    else:
        raise ValueError("raw_documents should be of type pd.Series, np.ndarray or list")


def _limit_features(X, vocabulary, high=None, low=None):
    """Remove too rare or too common features.

    Prune features that are non-zero in more documents than high or fewer
    documents than low, modifying the vocabulary.

    This does not prune samples with zero features.
    """
    if high is None and low is None:
        return X, set()

    # Calculate a mask based on document frequencies
    dfs = _document_frequency(X)
    mask = np.ones(len(dfs), dtype=bool)
    if high is not None:
        mask &= dfs <= high
    if low is not None:
        mask &= dfs >= low

    new_indices = np.cumsum(mask) - 1  # maps old indices to new
    for term, old_index in list(vocabulary.items()):
        if mask[old_index]:
            vocabulary[term] = int(new_indices[old_index])
        else:
            del vocabulary[term]
    kept_indices = np.where(mask)[0]
    if len(kept_indices) == 0:
        raise ValueError(
            "After pruning, no terms remain. Try a lower min_df or a higher max_df."
        )
    return X[:, kept_indices]


class FastCountVectorizer:
    def __init__(self, max_df=1.0, min_df=1, binary=False, dtype=np.float32, vocabulary=None):
        self.max_df = max_df
        self.min_df = min_df
        self.dtype = dtype
        self.binary = binary
        self.dfs = None
        if vocabulary is not None:
            if not isinstance(vocabulary, dict):
                raise ValueError(
                    "vocabulary should be a dict or None, got {!r}".format(vocabulary)
                )
            self.vocabulary_ = vocabulary
        else:
            self.vocabulary_ = None

    def fit(self, raw_documents):
        """Learn a vocabulary dictionary of all tokens in the raw documents.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which generates either str, unicode or file objects.

        Returns
        -------
        self : object
            Fitted vectorizer.
        """
        self.fit_transform(raw_documents)
        return self

    def fit_transform(self, raw_documents):
        """Learn the vocabulary dictionary and return document-term matrix.

        This is equivalent to fit followed by transform, but more efficiently
        implemented.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which generates either str, unicode or file objects.

        Returns
        -------
        X : array of shape (n_samples, n_features)
            Document-term matrix.
        """
        max_df = self.max_df
        min_df = self.min_df

        raw_documents = _check_iterable(raw_documents)

        values, j_indices, indptr, vocabulary = _count_vocab_cy(raw_documents, fixed_vocab=False)

        X = sp.csr_matrix(
            (values, j_indices, indptr),
            shape=(len(indptr) - 1, len(vocabulary)),
            dtype=self.dtype,
        )
        X.sum_duplicates()

        if self.binary:
            X.data.fill(1)

        if self.vocabulary_ is not None:
            warnings.warn("Overwriting vocabulary")
        n_doc = X.shape[0]
        max_doc_count = max_df if isinstance(max_df, int) else max_df * n_doc
        min_doc_count = min_df if isinstance(min_df, int) else min_df * n_doc
        if max_doc_count < min_doc_count:
            raise ValueError("max_df corresponds to < documents than min_df")
        X = _limit_features(X, vocabulary, max_doc_count, min_doc_count)
        self.vocabulary_ = vocabulary

        return X

    def partial_fit(self, raw_documents):
        """Incremental fit on a batch of documents.
        min_df and max_df are ignored during the partial_fit call.
        Call

        This method is expected to be called several times consecutively
        on different chunks of a dataset so as to implement incremental learning.

        Parameters
        ----------
        raw_documents : iterable
            An iterable of strings

        Returns
        -------
        self : object
            Fitted vectorizer.
        """
        raw_documents = _check_iterable(raw_documents)

        if self.vocabulary_ is None:
            self.vocabulary_ = {}

        values, j_indices, indptr, vocabulary = _count_vocab_cy(raw_documents, fixed_vocab=True,
                                                                vocabulary=self.vocabulary_, continue_fit=True)

        self.vocabulary_ = vocabulary

        # if we apply bincount to the raw j_indices, we count the number of times each word appears in the whole corpus
        # we want the number of documents each word appears in
        X = sp.csr_matrix(
            (values, j_indices, indptr),
            shape=(len(indptr) - 1, len(self.vocabulary_)),
            dtype=self.dtype,
        )
        X.sum_duplicates()

        new_dfs = np.bincount(X.indices, minlength=len(self.vocabulary_))

        if self.dfs is None:
            self.dfs = new_dfs
        else:
            # inplace resizing throws err during tests because of refcheck (pytest debugger references the array)
            self.dfs.resize(new_dfs.shape, refcheck=False)
            self.dfs += new_dfs

        return self

    def transform(self, raw_documents):
        """Transform documents to document-term matrix.

        Extract token counts out of raw text documents using the vocabulary
        fitted with fit or the one provided to the constructor.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which generates either str, unicode or file objects.

        Returns
        -------
        X : sparse matrix of shape (n_samples, n_features)
            Document-term matrix.
        """
        self._check_vocabulary()

        raw_documents = _check_iterable(raw_documents)
        # use the same matrix-building strategy as fit_transform
        values, j_indices, indptr, vocabulary = _count_vocab_cy(raw_documents, fixed_vocab=True,
                                                                vocabulary=self.vocabulary_, )
        X = sp.csr_matrix(
            (values, j_indices, indptr),
            shape=(len(indptr) - 1, len(vocabulary)),
            dtype=self.dtype,
        )
        X.sum_duplicates()
        if self.binary:
            X.data.fill(1)
        return X

    def _check_vocabulary(self):
        """Check if vocabulary is empty or missing (not fitted)"""
        if self.vocabulary_ is None:
            raise ValueError("Vocabulary not fitted or provided")

        if len(self.vocabulary_) == 0:
            raise ValueError("Vocabulary is empty")

    def limit_features(self):
        """Apply min_df after partial_fit"""
        if self.dfs is None:
            warnings.warn("Limiting features without prior fit or partial_fit, nothing will be done")
            return
        mask = np.ones(len(self.dfs), dtype=bool)
        mask &= self.dfs >= self.min_df
        if self.vocabulary_ is None:
            raise ValueError("Vocabulary not fitted")
        new_indices = np.cumsum(mask) - 1  # maps old indices to new
        for term, old_index in list(self.vocabulary_.items()):
            if mask[old_index]:
                self.vocabulary_[term] = int(new_indices[old_index])
            else:
                del self.vocabulary_[term]
