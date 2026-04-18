from collections.abc import Sequence
from typing import Any

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from qitrix.backend import resolve_backend
from qitrix.backend.adapters import to_sparse_array

from .types import Expression, NDArray, OperatorLike, SparseArray, SparseLike


def normalize_dims(dims: int | Sequence[int] | NDArray[Any]) -> list[int]:
    normalized_dims = np.reshape(dims, (-1,)).tolist()
    normalized_dims = [int(dim) for dim in normalized_dims]
    if any(dim < 1 for dim in normalized_dims):
        raise ValueError("dims must contain only positive integers")
    return normalized_dims


def as_sparse_array(op: SparseLike | NDArray[Any]) -> SparseArray:
    return to_sparse_array(op)


def identity_like(dim: int, template: Any) -> NDArray[Any] | SparseArray:
    if dim < 1:
        raise ValueError("identity dimension must be positive")
    return resolve_backend(template).eye(dim, template)


def shape_of(op: OperatorLike) -> tuple[int, ...]:
    shape = getattr(op, "shape", None)
    if shape is not None:
        return tuple(int(dim) for dim in shape)
    return resolve_backend(op).shape(op)


def normalize_axes(axes: int | Sequence[int], ndim: int) -> list[int]:
    if isinstance(axes, int):
        axes = [axes]

    normalized = []
    for axis in axes:
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise ValueError(f"axis {axis} is out of range for dims of length {ndim}")
        normalized.append(axis)

    if len(set(normalized)) != len(normalized):
        raise ValueError("axes must be unique")

    return normalized


def normalize_perm(perm: Sequence[int], ndim: int) -> list[int]:
    normalized = [int(axis) for axis in perm]
    expected = list(range(ndim))
    if sorted(normalized) != expected:
        raise ValueError(f"perm must be a permutation of {expected}")
    return normalized


def as_csr_array(op: Any) -> sp.csr_array:
    if isinstance(op, sp.csr_array):
        return op
    if sp.issparse(op):
        return sp.csr_array(op)
    if isinstance(op, cp.Expression):
        raise TypeError("cannot convert a CVXPY expression to csr_array")

    try:
        return sp.csr_array(np.asarray(op))
    except Exception as exc:
        raise TypeError("expected a matrix-like object") from exc


def validate_square_shape(op: OperatorLike, dims: Sequence[int]) -> None:
    shape = shape_of(op)
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"expected a square matrix, got shape {shape}")

    dim = int(np.prod(dims, dtype=int)) if dims else 1
    if shape != (dim, dim):
        raise ValueError(
            f"expected a square matrix of shape {(dim, dim)} for dims {list(dims)}, got {shape}"
        )


def trace_product(A: Any, B: Any):
    if isinstance(A, cp.Expression) or isinstance(B, cp.Expression):
        A_T = A.T if isinstance(A, cp.Expression) else np.asarray(A).T
        return cp.sum(cp.multiply(A_T, B))

    if sp.issparse(A) and sp.issparse(B):
        return A.multiply(B.T).sum()

    A_arr = A.toarray() if sp.issparse(A) else np.asarray(A)
    B_arr = B.toarray() if sp.issparse(B) else np.asarray(B)
    return np.sum(A_arr.T * B_arr)
