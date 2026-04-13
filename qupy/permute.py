from collections.abc import Sequence
from typing import Any, Literal, overload

import numpy as np
import scipy.sparse as sp

from ._types import Expression, NDArray, OperatorLike, SparseArray, SparseLike
from .utils import _as_sparse_array

def _permute_numpy(
    rho: NDArray[Any],
    dims: Sequence[int],
    perm: Sequence[int],
    direction: Literal["both", "left", "right"] = "both",
) -> NDArray[Any]:
    """
    Permute the axes of a density matrix using NumPy.

    Parameters
    ----------
    rho : NDArray[Any]
        Input operator written on the tensor-product space whose subsystem
        dimensions are given by ``dims``.
    dims : Sequence[int]
        Dimensions of the subsystems in the same order as the tensor factors.
    perm : Sequence[int]
        Permutation of subsystem indices. The indices are assumed to be already
        normalized to the range ``0, ..., len(dims) - 1``.
    direction : Literal["both", "left", "right"], optional
        Direction of permutation. Must be one of "both", "left", or "right".
        Default is "both".

    Returns
    -------
    NDArray[Any]
        The permuted operator.

    Notes
    -----
    The function reshapes ``rho`` into a tensor and applies ``np.transpose``
    to the subsystem indices selected by ``direction``.
    """
    dims = list(dims)
    perm = list(perm)
    dim = int(np.prod(dims, dtype=int)) if dims else 1
    if direction == "both":
        tensor_rho = np.asarray(rho).reshape(dims + dims)
        tensor_perm = perm + [p + len(dims) for p in perm]
        permuted_tensor_rho = np.transpose(tensor_rho, axes=tensor_perm)
        perm_rho = permuted_tensor_rho.reshape((dim, dim))
    elif direction == "left":
        tensor_rho = np.asarray(rho).reshape((*dims, -1))
        tensor_perm = perm + [len(dims)]
        permuted_tensor_rho = np.transpose(tensor_rho, axes=tensor_perm)
        perm_rho = permuted_tensor_rho.reshape((dim, -1))
    elif direction == "right":
        tensor_rho = np.asarray(rho).reshape((-1, *dims))
        tensor_perm = [0] + [p + 1 for p in perm]
        permuted_tensor_rho = np.transpose(tensor_rho, axes=tensor_perm)
        perm_rho = permuted_tensor_rho.reshape((-1, dim))
    else:
        raise ValueError("Invalid direction. Must be \"both\", \"left\", or \"right\".")

    return perm_rho

def _get_permutation_matrix(dims: Sequence[int], perm: Sequence[int]) -> sp.csr_array:
    """
    Construct the subsystem permutation matrix associated with ``perm``.

    Parameters
    ----------
    dims : Sequence[int]
        Dimensions of the subsystems.
    perm : Sequence[int]
        Target ordering of subsystem indices.

    Returns
    -------
    sp.csr_array
        Sparse permutation matrix acting on the vectorized Hilbert space.
    """
    d = np.prod(dims)
    coords = np.arange(d)
    ravel_coords = np.reshape(coords, dims)
    perm_coords = np.transpose(ravel_coords, perm).flatten()
    perm_matrix = sp.coo_array(([1.0] * d, (coords, perm_coords)), shape=(d, d)).tocsr()
    return perm_matrix

@overload
def permute(
    rho: Expression,
    dims: Sequence[int],
    perm: Sequence[int],
    direction: Literal["both", "left", "right"] = "both",
) -> Expression: ...


@overload
def permute(
    rho: SparseLike,
    dims: Sequence[int],
    perm: Sequence[int],
    direction: Literal["both", "left", "right"] = "both",
) -> SparseArray: ...


@overload
def permute(
    rho: NDArray[Any],
    dims: Sequence[int],
    perm: Sequence[int],
    direction: Literal["both", "left", "right"] = "both",
) -> NDArray[Any]: ...


def permute(
    rho: OperatorLike,
    dims: Sequence[int],
    perm: Sequence[int],
    direction: Literal["both", "left", "right"] = "both",
) -> OperatorLike:
    """
    Permute subsystem order in an operator or rectangular matrix.

    Parameters
    ----------
    rho : np.ndarray | scipy.sparse.spmatrix | sp.sparray | cvxpy.Expression
        Matrix-like input object.
    dims : Sequence[int]
        Dimensions of the subsystems being permuted.
    perm : Sequence[int]
        Target subsystem ordering.
    direction : Literal["both", "left", "right"], optional
        Which side of the matrix should be permuted. Supported values are
        ``"both"``, ``"left"``, and ``"right"``.

    Returns
    -------
    np.ndarray | sp.sparray | cvxpy.Expression
        Permuted matrix in the same backend family as ``rho``.
    """
    if isinstance(rho, np.ndarray):
        return _permute_numpy(rho, dims, perm, direction)
    if isinstance(rho, SparseLike):
        rho = _as_sparse_array(rho)

    perm_matrix = _get_permutation_matrix(dims, perm)

    if direction == "both":
        perm_rho = perm_matrix @ rho @ perm_matrix.T
    elif direction == "left":
        perm_rho = perm_matrix @ rho
    elif direction == "right":
        perm_rho = rho @ perm_matrix.T
    else:
        raise ValueError("Invalid direction. Must be \"both\", \"left\", or \"right\".")

    return perm_rho
