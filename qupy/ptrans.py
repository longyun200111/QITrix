from typing import Any

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from .tensor import tensor
from .permute import permute
from .utils import _as_csr_array, _normalize_axes


def _ptrans_numpy(rho: np.ndarray, dims: list[int], axes: list[int]) -> np.ndarray:
    """
    Compute the partial transpose of a dense matrix with NumPy tensor reshaping.

    Parameters
    ----------
    rho : np.ndarray
        Input operator written on the tensor-product space whose subsystem
        dimensions are given by ``dims``.
    dims : list[int]
        Dimensions of the subsystems in the same order as the tensor factors.
    axes : list[int]
        Subsystem indices to transpose. The indices are assumed to be already
        normalized to the range ``0, ..., len(dims) - 1``.

    Returns
    -------
    np.ndarray
        The partially transposed operator.

    Notes
    -----
    Writing ``rho`` as a tensor with index structure ``dims + dims``, partial
    transpose on subsystem ``k`` swaps the corresponding bra and ket indices.
    """
    dim = int(np.prod(dims, dtype=int)) if dims else 1
    tensor_rho = np.asarray(rho).reshape(dims + dims)
    perm = list(range(2 * len(dims)))
    for axis in axes:
        perm[axis], perm[axis + len(dims)] = perm[axis + len(dims)], perm[axis]
    perm_tensor_rho = np.transpose(tensor_rho, axes=perm)
    return perm_tensor_rho.reshape((dim, dim))


def _ptrans_cvxpy(rho: cp.Expression, dims: list[int], axes: list[int]) -> cp.Expression:
    """
    Compute the partial transpose of a CVXPY matrix expression.

    Parameters
    ----------
    rho : cp.Expression
        Matrix-valued CVXPY expression.
    dims : list[int]
        Dimensions of the tensor-product subsystems.
    axes : list[int]
        Subsystem indices to transpose.

    Returns
    -------
    cp.Expression
        CVXPY expression for the partially transposed operator.

    Notes
    -----
    The implementation uses a sparse superoperator acting on
    ``cp.vec(rho, order="F")``.
    """
    if not axes:
        return rho

    dim = int(np.prod(dims, dtype=int)) if dims else 1

    rho_vec = cp.vec(rho, order="F")
    perm = list(range(2 * len(dims)))
    for axis in axes:
        perm[axis], perm[axis + len(dims)] = perm[axis + len(dims)], perm[axis]
    perm_rho_vec = permute(
        rho_vec,
        dims=dims + dims,
        perm=perm,
        direction="left"
    )

    return cp.reshape(perm_rho_vec, (dim, dim), order="F")


def _ptrans_sum(rho: Any, dims: list[int], axes: list[int]) -> sp.csr_array:
    r"""
    Compute the partial transpose by the defining matrix-summation formula.

    Parameters
    ----------
    rho : Any
        Matrix-like input operator. The function first converts it to
        ``scipy.sparse.csr_array`` and then evaluates the defining summation.
    dims : list[int]
        Dimensions of the tensor-product subsystems.
    axes : list[int]
        Subsystem indices to transpose.

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
    if not axes:
        return _as_csr_array(rho)

    trans_rho = _as_csr_array(rho)

    for axis in axes:
        left_dim = int(np.prod(dims[:axis], dtype=int)) if axis > 0 else 1
        right_dim = int(np.prod(dims[axis + 1 :], dtype=int)) if axis + 1 < len(dims) else 1
        dim = int(np.prod(dims, dtype=int)) if dims else 1

        left_id = sp.eye_array(left_dim).tocsr()
        right_id = sp.eye_array(right_dim).tocsr()
        axis_dim = dims[axis]
        reduced_rho = sp.csr_array((dim, dim), dtype=trans_rho.dtype)

        for i in range(axis_dim):
            ket_i = sp.csr_array(([1.0], ([i], [0])), shape=(axis_dim, 1))
            bra_i = ket_i.T
            for j in range(axis_dim):
                ket_j = sp.csr_array(([1.0], ([j], [0])), shape=(axis_dim, 1))
                bra_j = ket_j.T
                basis_op = ket_j @ bra_i
                op = _as_csr_array(tensor(left_id, basis_op, right_id))
                reduced_rho = reduced_rho + _as_csr_array(op @ trans_rho @ op)

        trans_rho = reduced_rho

    return trans_rho


def ptrans(rho, dims, axes):
    """
    Compute the partial transpose of an operator over selected subsystems.

    Parameters
    ----------
    rho : np.ndarray | sp.sparray | cp.Expression
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

    - ``cvxpy.Expression`` inputs use a sparse superoperator acting on the
      vectorized matrix;
    - dense ``numpy.ndarray`` inputs use tensor reshaping and index swaps;
    - all remaining supported inputs fall back to the defining matrix
      summation with sparse matrix multiplications.
    """
    dims = np.reshape(dims, (-1,)).tolist()
    normalized_axes = _normalize_axes(axes, len(dims))

    if isinstance(rho, cp.Expression):
        return _ptrans_cvxpy(rho, dims, normalized_axes)
    if isinstance(rho, np.ndarray):
        return _ptrans_numpy(rho, dims, normalized_axes)
    return _ptrans_sum(rho, dims, normalized_axes)
