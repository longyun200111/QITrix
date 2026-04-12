from typing import Any, overload

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from .state import fock
from .permute import permute
from .tensor import tensor
from ._types import Expression, NDArray, OperatorLike, SparseArray, SparseLike
from .utils import _as_csr_array, _normalize_axes, _as_sparse_array


def _ptrans_numpy(
    rho: NDArray[Any], dims: list[int], axes: int | list[int]
) -> NDArray[Any]:
    """
    Compute the partial transpose of a dense matrix with NumPy tensor reshaping.

    Parameters
    ----------
    rho : np.ndarray
        Input operator written on the tensor-product space whose subsystem
        dimensions are given by ``dims``.
    dims : list[int]
        Dimensions of the subsystems in the same order as the tensor factors.
    axes : int | list[int]
        Subsystem indices to transpose. Negative indices are allowed and are
        normalized in the usual Python way.

    Returns
    -------
    np.ndarray
        The partially transposed operator.

    Notes
    -----
    Writing ``rho`` as a tensor with index structure ``dims + dims``, partial
    transpose on subsystem ``k`` swaps the corresponding bra and ket indices.
    """
    normalized_axes = _normalize_axes(axes, len(dims))
    dim = int(np.prod(dims, dtype=int)) if dims else 1
    tensor_rho = np.asarray(rho).reshape(dims + dims)
    perm = list(range(2 * len(dims)))
    for axis in normalized_axes:
        perm[axis], perm[axis + len(dims)] = perm[axis + len(dims)], perm[axis]
    perm_tensor_rho = np.transpose(tensor_rho, axes=perm)
    return perm_tensor_rho.reshape((dim, dim))


def _ptrans_cvxpy(
    rho: Expression, dims: list[int], axes: int | list[int]
) -> Expression:
    """
    Compute the partial transpose of a CVXPY matrix expression.

    Parameters
    ----------
    rho : cp.Expression
        Matrix-valued CVXPY expression.
    dims : list[int]
        Dimensions of the tensor-product subsystems.
    axes : int | list[int]
        Subsystem indices to transpose. Negative indices are allowed and are
        normalized in the usual Python way.

    Returns
    -------
    cp.Expression
        CVXPY expression for the partially transposed operator.

    Notes
    -----
    The implementation vectorizes ``rho`` in column-major order, permutes the
    doubled subsystem indices with ``permute(..., direction="left")``, and
    reshapes the result back to matrix form.
    """
    normalized_axes = _normalize_axes(axes, len(dims))
    if not normalized_axes:
        return rho

    dim = int(np.prod(dims, dtype=int)) if dims else 1

    rho_vec = cp.vec(rho, order="F")
    perm = list(range(2 * len(dims)))
    for axis in normalized_axes:
        perm[axis], perm[axis + len(dims)] = perm[axis + len(dims)], perm[axis]
    perm_rho_vec = permute(
        rho_vec,
        dims=dims + dims,
        perm=perm,
        direction="left"
    )

    return cp.reshape(perm_rho_vec, (dim, dim), order="F")


def _ptrans_sum(rho: NDArray[Any] | SparseLike, dims: list[int], axes: int | list[int]) -> SparseArray:
    r"""
    Compute the partial transpose by the defining matrix-summation formula.

    Parameters
    ----------
    rho : np.ndarray | scipy.sparse.spmatrix | sp.sparray
        Matrix-like input operator. The function first converts it to
        ``scipy.sparse.csr_array`` and then evaluates the defining summation.
    dims : list[int]
        Dimensions of the tensor-product subsystems.
    axes : int | list[int]
        Subsystem indices to transpose. Negative indices are allowed and are
        normalized in the usual Python way.

    Returns
    -------
    sp.csr_array
        Partially transposed operator in CSR format.

    Notes
    -----
    For one transposed subsystem, the implementation uses

    $$
    \rho^{T_T}
    =
    \sum_{i,j}
    \left(I \otimes |j\rangle\langle i| \otimes I\right)
    \rho
    \left(I \otimes |j\rangle\langle i| \otimes I\right).
    $$

    Multiple subsystem transposes are applied successively.
    """
    normalized_axes = _normalize_axes(axes, len(dims))
    if not normalized_axes:
        return _as_csr_array(rho)

    trans_rho = _as_csr_array(rho)

    for axis in normalized_axes:
        left_dim = int(np.prod(dims[:axis], dtype=int)) if axis > 0 else 1
        right_dim = int(np.prod(dims[axis + 1 :], dtype=int)) if axis + 1 < len(dims) else 1
        dim = int(np.prod(dims, dtype=int)) if dims else 1

        left_id = sp.eye_array(left_dim).tocsr()
        right_id = sp.eye_array(right_dim).tocsr()
        axis_dim = dims[axis]
        reduced_rho = sp.csr_array((dim, dim), dtype=trans_rho.dtype)

        for i in range(axis_dim):
            bra_i = fock(axis_dim, i, format="csr").T
            for j in range(axis_dim):
                ket_j = fock(axis_dim, j, format="csr")
                basis_op = ket_j @ bra_i
                op = _as_csr_array(tensor(left_id, basis_op, right_id))
                reduced_rho = reduced_rho + _as_csr_array(op @ trans_rho @ op)

        trans_rho = reduced_rho

    return trans_rho

@overload
def ptrans(
    rho: Expression, dims: int | list[int] | np.ndarray, axes: int | list[int]
) -> Expression: ...

@overload
def ptrans(
    rho: NDArray[Any], dims: int | list[int] | np.ndarray, axes: int | list[int]
) -> NDArray[Any]: ...

@overload
def ptrans(
    rho: SparseLike, dims: int | list[int] | np.ndarray, axes: int | list[int]
) -> SparseArray: ...


def ptrans(
    rho: OperatorLike, dims: int | list[int] | np.ndarray, axes: int | list[int]
) -> OperatorLike:
    """
    Compute the partial transpose of an operator over selected subsystems.

    Parameters
    ----------
    rho : np.ndarray | scipy.sparse.spmatrix | sp.sparray | cvxpy.Expression
        Input operator. Dense NumPy arrays and CVXPY matrix expressions have
        dedicated implementations. Any other supported matrix-like input is
        handled by the fallback summation-based branch.
    dims : int | list[int] | np.ndarray
        Subsystem dimensions. The total Hilbert-space dimension must satisfy
        ``np.prod(dims) == rho.shape[0] == rho.shape[1]``.
    axes : int | list[int]
        Subsystem indices to transpose. Negative indices are allowed and are
        normalized in the usual Python way.

    Returns
    -------
    np.ndarray | sp.csr_array | cp.Expression
        The partially transposed operator.

    Notes
    -----
    The dispatch is type-dependent:

    - ``cvxpy.Expression`` inputs vectorize the matrix, permute the doubled
      subsystem indices, and reshape back;
    - dense ``numpy.ndarray`` inputs use tensor reshaping and index swaps;
    - all remaining supported inputs fall back to the defining matrix
      summation with sparse matrix multiplications.
    """
    dims = np.reshape(dims, (-1,)).tolist()
    if isinstance(rho, cp.Expression):
        return _ptrans_cvxpy(rho, dims, axes)
    if isinstance(rho, np.ndarray):
        return _ptrans_numpy(rho, dims, axes)
    rho = _as_sparse_array(rho)
    return _ptrans_sum(rho, dims, axes)
