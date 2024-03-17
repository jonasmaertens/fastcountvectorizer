# _count_vocab_cy.pyx
import numpy as np
cimport numpy as np
from collections import defaultdict

def _count_vocab_cy(raw_documents, fixed_vocab, vocabulary=None, continue_fit=False):
    cdef int i
    cdef np.ndarray[int, ndim=1] j_indices
    cdef np.ndarray[int, ndim=1] indptr
    cdef np.ndarray[int, ndim=1] values

    if not fixed_vocab:
        vocabulary = defaultdict()
        vocabulary.default_factory = vocabulary.__len__

    if continue_fit:
        if vocabulary is None:
            raise ValueError(
                "vocabulary cannot be None when continue_fit is True"
            )
        # factory function of the defaultdict can only point to its own length after initialization
        # so we need to set the default_factory after the initialization, lambda as placeholder
        vocabulary = defaultdict(lambda: 0, vocabulary)
        vocabulary.default_factory = vocabulary.__len__

    # inplace splitting would reduce memory usage but would modify the raw_documents of the user
    split_docs = [doc.split() for doc in raw_documents]
    num = sum(len(doc) for doc in split_docs)
    j_indices = np.zeros(num, dtype=np.int32)
    indptr = np.zeros(len(raw_documents)+1, dtype=np.int32)
    i = 0
    for idx, doc in enumerate(split_docs):
        for feature in doc:
            try:
                j_indices[i] = vocabulary[feature]
                i += 1
            except KeyError:
                continue
        indptr[idx+1] = i
    values = np.ones(num, dtype=np.int32)

    if not fixed_vocab or continue_fit:
        vocabulary = dict(vocabulary)
        if not vocabulary:
            raise ValueError(
                "empty vocabulary; perhaps the documents only contain stop words"
            )
    return values, j_indices, indptr, vocabulary