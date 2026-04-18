from collections.abc import Sequence
from typing import Any, Literal, overload

import numpy as np
import scipy.sparse as sp

from qitrix.backend import resolve_backend
from qitrix._internal.types import Expression, NDArray, OperatorLike, SparseArray, SparseLike
from qitrix._internal.utils import as_sparse_array, normalize_dims, normalize_perm, shape_of
from qitrix.core.operator import Operator


def _permute_numpy(
    rho: NDArray[Any],
    dims: Sequence[int],
    perm: Sequence[int],
    direction: Literal["both", "left", "right"] = "both",
) -> NDArray[Any]:
    dims = list(dims)
    perm = list(perm)
    dim = int(np.prod(dims, dtype=int)) if dims else 1
    if direction == "both":
        tensor_rho = np.asarray(rho).reshape(dims + dims)
        tensor_perm = perm + [p + len(dims) for p in perm]
        permuted_tensor_rho = np.transpose(tensor_rho, axes=tensor_perm)
        return permuted_tensor_rho.reshape((dim, dim))
    if direction == "left":
        tensor_rho = np.asarray(rho).reshape((*dims, -1))
        tensor_perm = perm + [len(dims)]
        permuted_tensor_rho = np.transpose(tensor_rho, axes=tensor_perm)
        return permuted_tensor_rho.reshape((dim, -1))
    if direction == "right":
        tensor_rho = np.asarray(rho).reshape((-1, *dims))
        tensor_perm = [0] + [p + 1 for p in perm]
        permuted_tensor_rho = np.transpose(tensor_rho, axes=tensor_perm)
        return permuted_tensor_rho.reshape((-1, dim))
    raise ValueError('Invalid direction. Must be "both", "left", or "right".')


def _permute_operator(
    rho: Operator,
    labels: Sequence[str],
    direction: Literal["both", "left", "right"] = "both",
) -> Operator:
    if direction == "both":
        input_space = rho.input_space.reordered(labels)
        output_space = rho.output_space.reordered(labels)
        data = permute(rho.data, rho.input_space.dims, rho.input_space.perm(labels), direction)
    elif direction == "left":
        output_space = rho.output_space.reordered(labels)
        input_space = rho.input_space
        data = permute(rho.data, rho.output_space.dims, rho.output_space.perm(labels), direction)
    elif direction == "right":
        input_space = rho.input_space.reordered(labels)
        output_space = rho.output_space
        data = permute(rho.data, rho.input_space.dims, rho.input_space.perm(labels), direction)
    else:
        raise ValueError('Invalid direction. Must be "both", "left", or "right".')
    return Operator(data, input_space=input_space, output_space=output_space)


def _get_permutation_matrix(dims: Sequence[int], perm: Sequence[int]) -> sp.csr_array:
    d = np.prod(dims)
    coords = np.arange(d)
    ravel_coords = np.reshape(coords, dims)
    perm_coords = np.transpose(ravel_coords, perm).flatten()
    return sp.coo_array(([1.0] * d, (coords, perm_coords)), shape=(d, d)).tocsr()


@overload
def permute(rho: Operator, dims: Sequence[str], perm: None = None, direction: Literal["both", "left", "right"] = "both") -> Operator: ...


@overload
def permute(rho: Expression, dims: Sequence[int], perm: Sequence[int], direction: Literal["both", "left", "right"] = "both") -> Expression: ...


@overload
def permute(rho: SparseLike, dims: Sequence[int], perm: Sequence[int], direction: Literal["both", "left", "right"] = "both") -> SparseArray: ...


@overload
def permute(rho: NDArray[Any], dims: Sequence[int], perm: Sequence[int], direction: Literal["both", "left", "right"] = "both") -> NDArray[Any]: ...


def permute(
    rho: OperatorLike | Operator,
    dims: Sequence[int] | Sequence[str],
    perm: Sequence[int] | None = None,
    direction: Literal["both", "left", "right"] = "both",
) -> OperatorLike | Operator:
    if isinstance(rho, Operator):
        if perm is not None:
            raise ValueError("Operator permute expects reordered subsystem labels as the second argument")
        return _permute_operator(rho, dims, direction)
    if perm is None:
        raise ValueError("perm must be provided for raw matrix-like inputs")

    dims = normalize_dims(dims)
    perm = normalize_perm(perm, len(dims))
    dim = int(np.prod(dims, dtype=int)) if dims else 1
    shape = shape_of(rho)

    if len(shape) == 1:
        if direction != "left":
            raise ValueError(f"expected a 2D matrix for direction='{direction}', got shape {shape}")
        if shape[0] != dim:
            raise ValueError(
                f"expected a vector of length {dim} for direction='left' with dims {dims}, got shape {shape}"
            )
    elif len(shape) != 2:
        raise ValueError(f"expected a matrix-like object, got shape {shape}")
    if direction == "both" and shape != (dim, dim):
        raise ValueError(f"expected shape {(dim, dim)} for direction='both' with dims {dims}, got {shape}")
    if len(shape) == 2 and direction == "left" and shape[0] != dim:
        raise ValueError(f"expected {dim} rows for direction='left' with dims {dims}, got shape {shape}")
    if len(shape) == 2 and direction == "right" and shape[1] != dim:
        raise ValueError(f"expected {dim} columns for direction='right' with dims {dims}, got shape {shape}")

    backend = resolve_backend(rho)
    if backend.kind == "dense":
        return _permute_numpy(rho, dims, perm, direction)
    if backend.kind == "sparse":
        rho = as_sparse_array(rho)

    perm_matrix = _get_permutation_matrix(dims, perm)
    if direction == "both":
        return perm_matrix @ rho @ perm_matrix.T
    if direction == "left":
        return perm_matrix @ rho
    return rho @ perm_matrix.T
