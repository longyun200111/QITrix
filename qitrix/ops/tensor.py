from typing import Any, overload

import scipy.sparse as sp

from qitrix.backend import resolve_backend
from qitrix.backend.adapters import to_sparse_array
from qitrix._internal.types import (
    DenseArrayT,
    Expression,
    NDArray,
    NonCvxOperatorLike,
    OperatorLike,
    SparseArray,
    SparseLike,
)


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
    backend = resolve_backend(left, right)
    if backend.kind == "sparse":
        if sp.issparse(left):
            left = to_sparse_array(left).asformat("coo")
        if sp.issparse(right):
            right = to_sparse_array(right).asformat("coo")
    return backend.kron(left, right)


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
def tensor(__first: SparseLike, *factors: NonCvxOperatorLike) -> sp.bsr_array | sp.coo_array: ...


@overload
def tensor(
    __first: NonCvxOperatorLike,
    __second: SparseLike,
    *factors: NonCvxOperatorLike,
) -> sp.bsr_array | sp.coo_array: ...


@overload
def tensor(__first: NonCvxOperatorLike, __second: NonCvxOperatorLike, __third: SparseLike, *factors: NonCvxOperatorLike) -> sp.bsr_array | sp.coo_array: ...


def tensor(*factors: OperatorLike) -> OperatorLike:
    if not factors:
        raise ValueError("at least one factor is required")
    result = factors[0]
    for factor in factors[1:]:
        result = _kron(result, factor)
    return result
