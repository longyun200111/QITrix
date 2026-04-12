"""
Shared typing declarations for the ``qupy`` package.

This module centralizes the type aliases and type variables used across the
public linear-algebra helpers.
"""

from typing import Any, TypeVar
from numpy.typing import NDArray
from cvxpy import Expression

import scipy.sparse as sp

SparseArray = (
    sp.bsr_array
    | sp.coo_array
    | sp.csc_array
    | sp.csr_array
    | sp.dia_array
    | sp.dok_array
    | sp.lil_array
)
SparseMatrix = (
    sp.bsr_matrix
    | sp.coo_matrix
    | sp.csc_matrix
    | sp.csr_matrix
    | sp.dia_matrix
    | sp.dok_matrix
    | sp.lil_matrix
)
SparseLike = SparseArray | SparseMatrix
NonCvxOperatorLike = NDArray[Any] | SparseLike
OperatorLike = NDArray[Any] | SparseLike | Expression

DenseArrayT = TypeVar("DenseArrayT", bound=NDArray[Any])
SparseArrayT = TypeVar("SparseArrayT", bound=SparseArray)
CvxArrayT = TypeVar("CvxArrayT", bound=Expression)
OperatorLikeT = TypeVar("OperatorLikeT", bound=OperatorLike)


__all__ = [
    "NDArray",
    "SparseArray",
    "SparseMatrix",
    "SparseLike",
    "Expression",
    "NonCvxOperatorLike",
    "OperatorLike",
    "DenseArrayT",
    "SparseArrayT",
    "CvxArrayT",
    "OperatorLikeT",
]
