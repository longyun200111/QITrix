from typing import Any, overload

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from ._types import (
    DenseArrayT,
    Expression,
    NDArray,
    NonCvxOperatorLike,
    OperatorLike,
    SparseArray,
    SparseLike,
)
from .utils import _as_sparse_array

@overload
def _kron(left: Expression, right: OperatorLike) -> Expression: ...

@overload
def _kron(left: OperatorLike, right: Expression) -> Expression: ...

@overload
def _kron(left: DenseArrayT, right: DenseArrayT) -> DenseArrayT: ...

@overload
def _kron(left: NonCvxOperatorLike, right: SparseLike) -> sp.bsr_array | sp.coo_array: ...

@overload
def _kron(left: SparseLike, right: NonCvxOperatorLike) -> sp.bsr_array | sp.coo_array: ...

def _kron(left: OperatorLike, right: OperatorLike) -> OperatorLike:
    """
    Compute a Kronecker product while preserving the most expressive type.

    Parameters
    ----------
    left, right : Any
        Matrix-like operands. Dense NumPy arrays, SciPy sparse matrices/arrays,
        and CVXPY expressions are supported.

    Returns
    -------
    np.ndarray | sp.sparray | cp.Expression
        Kronecker product of ``left`` and ``right``.

    Notes
    -----
    If either operand is a CVXPY expression, the result is built with
    ``cp.kron``. Otherwise, sparse operands are handled with ``sp.kron`` and
    dense operands with ``np.kron``.
    """
    if isinstance(left, cp.Expression) or isinstance(right, cp.Expression):
        return cp.kron(left, right)
    if sp.issparse(left) or sp.issparse(right):
        if sp.issparse(left):
            left = _as_sparse_array(left).asformat("coo")
        if sp.issparse(right):
            right = _as_sparse_array(right).asformat("coo")
        return _as_sparse_array(sp.kron(left, right))
    return np.kron(np.asarray(left), np.asarray(right))


@overload
def tensor(__first: Expression, *factors: OperatorLike) -> Expression: ...

@overload
def tensor(__first: OperatorLike, __second: Expression, *factors: OperatorLike) -> Expression: ...

@overload
def tensor(__first: OperatorLike, __second: OperatorLike, __third: Expression, *factors: OperatorLike) -> Expression: ...

@overload
def tensor(*factors: NDArray[Any]) -> NDArray[Any]: ...

@overload
def tensor(*factors: SparseArray) -> sp.bsr_array | sp.coo_array: ...

@overload
def tensor(
    __first: SparseLike, *factors: NonCvxOperatorLike
) -> sp.bsr_array | sp.coo_array: ...

@overload
def tensor(
    __first: NonCvxOperatorLike,
    __second: SparseLike,
    *factors: NonCvxOperatorLike,
) -> sp.bsr_array | sp.coo_array: ...

@overload
def tensor(__first: NonCvxOperatorLike, __second: NonCvxOperatorLike, __third: SparseLike, *factors: NonCvxOperatorLike) -> sp.bsr_array | sp.coo_array: ...

def tensor(*factors: OperatorLike) -> OperatorLike:
    """
    Compute the tensor product of multiple factors from left to right.

    Parameters
    ----------
    *factors : tuple[Any, ...]
        Sequence of matrix-like factors to be tensored together.

    Returns
    -------
    np.ndarray | sp.sparray | cp.Expression
        Tensor product of all supplied factors.

    Raises
    ------
    ValueError
        If no factor is provided.
    """
    if not factors:
        raise ValueError("at least one factor is required")
    result = factors[0]
    for factor in factors[1:]:
        result = _kron(result, factor)
    return result
