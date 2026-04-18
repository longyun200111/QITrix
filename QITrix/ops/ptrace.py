from collections.abc import Sequence
from typing import Any, overload

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from QITrix.backend import resolve_backend
from QITrix._internal.types import Expression, NDArray, SparseArray, SparseLike
from QITrix._internal.utils import (
    as_csr_array,
    as_sparse_array,
    normalize_axes,
    normalize_dims,
    validate_square_shape,
)
from QITrix.core.operator import Operator

from .extend import extend
from .state import fock
from .tensor import tensor


def _ptrace_numpy(rho: NDArray[Any], dims: Sequence[int], axes: int | Sequence[int]) -> NDArray[Any]:
    dims = list(dims)
    normalized_axes = normalize_axes(axes, len(dims))
    tensor_rho = np.asarray(rho).reshape(dims + dims)
    current_dims = dims.copy()

    for axis in sorted(normalized_axes, reverse=True):
        tensor_rho = np.trace(tensor_rho, axis1=axis, axis2=axis + len(current_dims))
        del current_dims[axis]

    out_dim = int(np.prod(current_dims)) if current_dims else 1
    return tensor_rho.reshape((out_dim, out_dim))


def _partial_trace_superop(dims: Sequence[int], axes: int | Sequence[int]) -> sp.csr_array:
    dims = list(dims)
    normalized_axes = normalize_axes(axes, len(dims))
    trace_dim = int(np.prod([dims[axis] for axis in normalized_axes], dtype=int))
    remaining_dims = [dim for i, dim in enumerate(dims) if i not in normalized_axes]

    maximal_ent = sp.eye_array(trace_dim).reshape((1, -1)).tocsr()
    superop = extend(
        maximal_ent,
        input_dims=dims + dims,
        output_dims=remaining_dims + remaining_dims,
        input_axes=normalized_axes + [axis + len(dims) for axis in normalized_axes],
        output_axes=[],
    )
    return as_csr_array(superop)


def _ptrace_cvxpy(rho: Expression, dims: Sequence[int], axes: int | Sequence[int]) -> Expression:
    dims = list(dims)
    normalized_axes = normalize_axes(axes, len(dims))
    if not normalized_axes:
        return rho

    superop = _partial_trace_superop(dims, normalized_axes)
    out_dims = [dim for i, dim in enumerate(dims) if i not in normalized_axes]
    out_dim = int(np.prod(out_dims, dtype=int)) if out_dims else 1
    return cp.reshape(superop @ cp.vec(rho, order="F"), (out_dim, out_dim), order="F")


def _ptrace_sum(rho, dims: Sequence[int], axes: int | Sequence[int]) -> SparseArray:
    dims = list(dims)
    normalized_axes = normalize_axes(axes, len(dims))
    if not normalized_axes:
        return as_csr_array(rho)

    traced_rho = as_csr_array(rho)
    current_dims = dims.copy()
    for axis in sorted(normalized_axes, reverse=True):
        left_dim = int(np.prod(current_dims[:axis], dtype=int)) if axis > 0 else 1
        right_dim = int(np.prod(current_dims[axis + 1 :], dtype=int)) if axis + 1 < len(current_dims) else 1
        out_dims = current_dims[:axis] + current_dims[axis + 1 :]
        out_dim = int(np.prod(out_dims, dtype=int)) if out_dims else 1

        left_id = sp.eye_array(left_dim).tocsr()
        right_id = sp.eye_array(right_dim).tocsr()
        reduced_rho = sp.csr_array((out_dim, out_dim), dtype=traced_rho.dtype)

        for i in range(current_dims[axis]):
            basis = fock(current_dims[axis], i, format="csr")
            left = as_csr_array(tensor(left_id, basis.T, right_id))
            right = as_csr_array(tensor(left_id, basis, right_id))
            reduced_rho = reduced_rho + as_csr_array(left @ traced_rho @ right)

        traced_rho = reduced_rho
        current_dims = out_dims

    return traced_rho


@overload
def ptrace(rho: Operator, dims: str | Sequence[str], axes: None = None) -> Operator: ...


@overload
def ptrace(rho: Expression, dims: int | Sequence[int] | np.ndarray, axes: int | Sequence[int]) -> Expression: ...


@overload
def ptrace(rho: NDArray[Any], dims: int | Sequence[int] | np.ndarray, axes: int | Sequence[int]) -> NDArray[Any]: ...


@overload
def ptrace(rho: SparseLike, dims: int | Sequence[int] | np.ndarray, axes: int | Sequence[int]) -> SparseArray: ...


def ptrace(rho, dims, axes: int | Sequence[int] | None = None):
    if isinstance(rho, Operator):
        if axes is not None:
            raise ValueError("Operator ptrace expects traced subsystem labels as the second argument")
        space = rho.space
        traced_axes = space.indices(dims)
        reduced = ptrace(rho.data, space.dims, traced_axes)
        return Operator(reduced, space=space.drop(dims))

    if axes is None:
        raise ValueError("axes must be provided for raw matrix-like inputs")

    dims = normalize_dims(dims)
    validate_square_shape(rho, dims)
    backend = resolve_backend(rho)
    if backend.kind == "symbolic":
        return _ptrace_cvxpy(rho, dims, axes)
    if backend.kind == "dense":
        return _ptrace_numpy(rho, dims, axes)
    return _ptrace_sum(as_sparse_array(rho), dims, axes)
