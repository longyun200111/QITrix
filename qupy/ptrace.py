from collections.abc import Sequence
from typing import Any, overload

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from .extend import extend
from .state import fock
from .tensor import tensor
from .utils import _normalize_axes, _as_csr_array, _as_sparse_array
from ._types import Expression, NDArray, OperatorLike, SparseArray, SparseLike


def _ptrace_numpy(
    rho: NDArray[Any], dims: Sequence[int], axes: int | Sequence[int]
) -> NDArray[Any]:
    """
    Compute the partial trace of a dense matrix with NumPy tensor reshaping.

    Parameters
    ----------
    rho : np.ndarray
        Input operator written on the tensor-product space whose subsystem
        dimensions are given by ``dims``.
    dims : Sequence[int]
        Dimensions of the subsystems in the same order as the tensor factors.
    axes : int | Sequence[int]
        Subsystem indices to trace out. Negative indices are allowed and are
        normalized in the usual Python way.

    Returns
    -------
    np.ndarray
        The reduced operator after tracing out the subsystems in ``axes``.

    Notes
    -----
    This implementation reshapes ``rho`` into a tensor with index structure
    ``dims + dims`` and then repeatedly applies ``np.trace`` to the bra/ket
    index pair associated with each traced subsystem.
    """
    dims = list(dims)
    normalized_axes = _normalize_axes(axes, len(dims))
    tensor = np.asarray(rho).reshape(dims + dims)
    current_dims = dims.copy()

    for axis in sorted(normalized_axes, reverse=True):
        tensor = np.trace(tensor, axis1=axis, axis2=axis + len(current_dims))
        del current_dims[axis]

    out_dim = int(np.prod(current_dims)) if current_dims else 1
    return tensor.reshape((out_dim, out_dim))


def _partial_trace_superop(dims: Sequence[int], axes: int | Sequence[int]) -> sp.csr_array:
    """
    Construct the sparse superoperator representing a partial trace map.

    Parameters
    ----------
    dims : Sequence[int]
        Dimensions of the input subsystems.
    axes : int | Sequence[int]
        Subsystem indices to trace out. Negative indices are allowed and are
        normalized in the usual Python way.

    Returns
    -------
    sp.csr_array
        Sparse matrix ``S`` such that
        ``vec(Tr_axes(rho)) = S @ vec(rho)``
        in column-major order.

    Notes
    -----
    This helper keeps the existing construction based on extending the
    vectorized maximally entangled effect to the full doubled Hilbert space.
    It is used by the CVXPY branch, where the linear map form is convenient.
    """
    dims = list(dims)
    normalized_axes = _normalize_axes(axes, len(dims))
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
    return _as_csr_array(superop)


def _ptrace_cvxpy(rho: Expression, dims: Sequence[int], axes: int | Sequence[int]) -> Expression:
    """
    Compute the partial trace of a CVXPY matrix expression.

    Parameters
    ----------
    rho : cp.Expression
        Matrix-valued CVXPY expression.
    dims : Sequence[int]
        Dimensions of the tensor-product subsystems.
    axes : int | Sequence[int]
        Subsystem indices to trace out. Negative indices are allowed and are
        normalized in the usual Python way.

    Returns
    -------
    cp.Expression
        CVXPY expression for the reduced operator.

    Notes
    -----
    The implementation uses the sparse superoperator returned by
    ``_partial_trace_superop`` and applies it to ``cp.vec(rho, order="F")``.
    """
    dims = list(dims)
    normalized_axes = _normalize_axes(axes, len(dims))
    if not normalized_axes:
        return rho

    superop = _partial_trace_superop(dims, normalized_axes)
    out_dims = [dim for i, dim in enumerate(dims) if i not in normalized_axes]
    out_dim = int(np.prod(out_dims, dtype=int)) if out_dims else 1

    return cp.reshape(
        superop @ cp.vec(rho, order="F"),
        (out_dim, out_dim),
        order="F",
    )

def _ptrace_sum(rho: OperatorLike, dims: Sequence[int], axes: int | Sequence[int]) -> OperatorLike:
    r"""
    Compute the partial trace of a sparse operator by the defining summation.

    Parameters
    ----------
    rho : OperatorLike
        Matrix-like input operator. The function first converts it to
        ``scipy.sparse.csr_array`` and then evaluates the defining summation.
    dims : Sequence[int]
        Dimensions of the tensor-product subsystems.
    axes : int | Sequence[int]
        Subsystem indices to trace out. Negative indices are allowed and are
        normalized in the usual Python way.

    Returns
    -------
    sp.csr_array
        Reduced sparse operator in CSR format.

    Notes
    -----
    The implementation follows the defining formula

    $$
    \mathrm{Tr}_{T}(\rho)
    =
    \sum_{\boldsymbol{i}}
    \left(I \otimes \langle \boldsymbol{i}|\right)
    \rho
    \left(I \otimes |\boldsymbol{i}\rangle\right),
    $$

    where the sum runs over the basis indices of the traced subsystems.
    """
    dims = list(dims)
    normalized_axes = _normalize_axes(axes, len(dims))
    if not normalized_axes:
        return _as_csr_array(rho)

    traced_rho = _as_csr_array(rho)
    current_dims = dims.copy()

    for axis in sorted(normalized_axes, reverse=True):
        left_dim = int(np.prod(current_dims[:axis], dtype=int)) if axis > 0 else 1
        right_dim = (
            int(np.prod(current_dims[axis + 1 :], dtype=int))
            if axis + 1 < len(current_dims)
            else 1
        )
        out_dims = current_dims[:axis] + current_dims[axis + 1 :]
        out_dim = int(np.prod(out_dims, dtype=int)) if out_dims else 1

        left_id = sp.eye_array(left_dim).tocsr()
        right_id = sp.eye_array(right_dim).tocsr()
        traced_axis_dim = current_dims[axis]
        reduced_rho = sp.csr_array((out_dim, out_dim), dtype=traced_rho.dtype)

        for i in range(traced_axis_dim):
            basis = fock(traced_axis_dim, i, format="csr")
            left = _as_csr_array(tensor(left_id, basis.T, right_id))
            right = _as_csr_array(tensor(left_id, basis, right_id))
            reduced_rho = reduced_rho + _as_csr_array(left @ traced_rho @ right)

        traced_rho = reduced_rho
        current_dims = out_dims

    return traced_rho


@overload
def ptrace(
    rho: Expression, dims: int | Sequence[int] | np.ndarray, axes: int | Sequence[int]
) -> Expression: ...


@overload
def ptrace(
    rho: NDArray[Any], dims: int | Sequence[int] | np.ndarray, axes: int | Sequence[int]
) -> NDArray[Any]: ...


@overload
def ptrace(
    rho: SparseLike, dims: int | Sequence[int] | np.ndarray, axes: int | Sequence[int]
) -> SparseArray: ...


def ptrace(
    rho: NDArray[Any] | SparseLike | Expression,
    dims: int | Sequence[int] | np.ndarray,
    axes: int | Sequence[int],
) -> NDArray[Any] | SparseArray | Expression:
    """
    Compute the partial trace of an operator over selected subsystems.

    Parameters
    ----------
    rho : np.ndarray | scipy.sparse.spmatrix | sp.sparray | cvxpy.Expression
        Input operator. Dense NumPy arrays and CVXPY matrix expressions have
        dedicated implementations. Any other supported matrix-like input is
        handled by the fallback summation-based branch.
    dims : int | Sequence[int] | np.ndarray
        Subsystem dimensions. The total Hilbert-space dimension must satisfy
        ``np.prod(dims) == rho.shape[0] == rho.shape[1]``.
    axes : int | Sequence[int]
        Subsystem indices to trace out. Negative indices are allowed and are
        normalized in the usual Python way.

    Returns
    -------
    np.ndarray | sp.csr_array | cp.Expression
        The reduced operator after tracing out the subsystems in ``axes``.

    Notes
    -----
    The dispatch is type-dependent:

    - ``cvxpy.Expression`` inputs use a sparse superoperator acting on the
      vectorized matrix;
    - dense ``numpy.ndarray`` inputs use tensor reshaping and ``np.trace``;
    - all remaining supported inputs fall back to the defining summation with
      sparse matrix multiplications.
    """
    dims = np.reshape(dims, (-1,)).tolist()
    if isinstance(rho, cp.Expression):
        return _ptrace_cvxpy(rho, dims, axes)
    if isinstance(rho, np.ndarray):
        return _ptrace_numpy(rho, dims, axes)
    rho = _as_sparse_array(rho)
    return _ptrace_sum(rho, dims, axes)
