from collections.abc import Sequence
from typing import Any, overload

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from QITrix.backend import resolve_backend
from QITrix._internal.types import Expression, NDArray, OperatorLike, SparseArray, SparseLike
from QITrix._internal.utils import as_csr_array, as_sparse_array, normalize_axes, normalize_dims, validate_square_shape
from QITrix.core.operator import Operator

from .permute import permute
from .state import fock
from .tensor import tensor


def _ptrans_numpy(rho: NDArray[Any], dims: Sequence[int], axes: int | Sequence[int]) -> NDArray[Any]:
    dims = list(dims)
    normalized_axes = normalize_axes(axes, len(dims))
    dim = int(np.prod(dims, dtype=int)) if dims else 1
    tensor_rho = np.asarray(rho).reshape(dims + dims)
    perm = list(range(2 * len(dims)))
    for axis in normalized_axes:
        perm[axis], perm[axis + len(dims)] = perm[axis + len(dims)], perm[axis]
    return np.transpose(tensor_rho, axes=perm).reshape((dim, dim))


def _ptrans_cvxpy(rho: Expression, dims: Sequence[int], axes: int | Sequence[int]) -> Expression:
    dims = list(dims)
    normalized_axes = normalize_axes(axes, len(dims))
    if not normalized_axes:
        return rho

    dim = int(np.prod(dims, dtype=int)) if dims else 1
    rho_vec = cp.vec(rho, order="F")
    perm = list(range(2 * len(dims)))
    for axis in normalized_axes:
        perm[axis], perm[axis + len(dims)] = perm[axis + len(dims)], perm[axis]
    perm_rho_vec = permute(rho_vec, dims=dims + dims, perm=perm, direction="left")
    return cp.reshape(perm_rho_vec, (dim, dim), order="F")


def _ptrans_sum(rho: NDArray[Any] | SparseLike, dims: Sequence[int], axes: int | Sequence[int]) -> SparseArray:
    dims = list(dims)
    normalized_axes = normalize_axes(axes, len(dims))
    if not normalized_axes:
        return as_csr_array(rho)

    trans_rho = as_csr_array(rho)
    for axis in normalized_axes:
        left_dim = int(np.prod(dims[:axis], dtype=int)) if axis > 0 else 1
        right_dim = int(np.prod(dims[axis + 1 :], dtype=int)) if axis + 1 < len(dims) else 1
        dim = int(np.prod(dims, dtype=int)) if dims else 1
        left_id = sp.eye_array(left_dim).tocsr()
        right_id = sp.eye_array(right_dim).tocsr()
        reduced_rho = sp.csr_array((dim, dim), dtype=trans_rho.dtype)

        for i in range(dims[axis]):
            bra_i = fock(dims[axis], i, format="csr").T
            for j in range(dims[axis]):
                ket_j = fock(dims[axis], j, format="csr")
                basis_op = ket_j @ bra_i
                op = as_csr_array(tensor(left_id, basis_op, right_id))
                reduced_rho = reduced_rho + as_csr_array(op @ trans_rho @ op)
        trans_rho = reduced_rho
    return trans_rho


@overload
def ptrans(rho: Operator, dims: str | Sequence[str], axes: None = None) -> Operator: ...


@overload
def ptrans(rho: Expression, dims: int | Sequence[int] | np.ndarray, axes: int | Sequence[int]) -> Expression: ...


@overload
def ptrans(rho: NDArray[Any], dims: int | Sequence[int] | np.ndarray, axes: int | Sequence[int]) -> NDArray[Any]: ...


@overload
def ptrans(rho: SparseLike, dims: int | Sequence[int] | np.ndarray, axes: int | Sequence[int]) -> SparseArray: ...


def ptrans(rho: OperatorLike | Operator, dims, axes: int | Sequence[int] | None = None):
    if isinstance(rho, Operator):
        if axes is not None:
            raise ValueError("Operator ptrans expects subsystem labels as the second argument")
        transformed = ptrans(rho.data, rho.space.dims, rho.space.indices(dims))
        return Operator(transformed, space=rho.space)

    if axes is None:
        raise ValueError("axes must be provided for raw matrix-like inputs")
    dims = normalize_dims(dims)
    validate_square_shape(rho, dims)
    backend = resolve_backend(rho)
    if backend.kind == "symbolic":
        return _ptrans_cvxpy(rho, dims, axes)
    if backend.kind == "dense":
        return _ptrans_numpy(rho, dims, axes)
    return _ptrans_sum(as_sparse_array(rho), dims, axes)
