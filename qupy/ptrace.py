from typing import Any

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from .extend import extend
from .state import fock
from .tensor import tensor
from .utils import _normalize_axes, _as_csr_array


def _ptrace_numpy(rho: np.ndarray, dims: list[int], axes: list[int]) -> np.ndarray:
    """
    Compute the partial trace of a dense matrix with NumPy tensor reshaping.

    Parameters
    ----------
    rho : np.ndarray
        Input operator written on the tensor-product space whose subsystem
        dimensions are given by ``dims``.
    dims : list[int]
        Dimensions of the subsystems in the same order as the tensor factors.
    axes : list[int]
        Subsystem indices to trace out. The indices are assumed to be already
        normalized to the range ``0, ..., len(dims) - 1``.

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
    tensor = np.asarray(rho).reshape(dims + dims)
    current_dims = dims.copy()

    for axis in sorted(axes, reverse=True):
        tensor = np.trace(tensor, axis1=axis, axis2=axis + len(current_dims))
        del current_dims[axis]

    out_dim = int(np.prod(current_dims)) if current_dims else 1
    return tensor.reshape((out_dim, out_dim))


def _partial_trace_superop(dims: list[int], axes: list[int]) -> sp.csr_array:
    """
    Construct the sparse superoperator representing a partial trace map.

    Parameters
    ----------
    dims : list[int]
        Dimensions of the input subsystems.
    axes : list[int]
        Subsystem indices to trace out.

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
    trace_dim = int(np.prod([dims[axis] for axis in axes], dtype=int))
    remaining_dims = [dim for i, dim in enumerate(dims) if i not in axes]

    maximal_ent = sp.eye_array(trace_dim).reshape((1, -1)).tocsr()
    superop = extend(
        maximal_ent,
        input_dims=dims + dims,
        output_dims=remaining_dims + remaining_dims,
        input_axes=axes + [axis + len(dims) for axis in axes],
        output_axes=[],
    )
    return _as_csr_array(superop)


def _ptrace_cvxpy(rho: cp.Expression, dims: list[int], axes: list[int]) -> cp.Expression:
    """
    Compute the partial trace of a CVXPY matrix expression.

    Parameters
    ----------
    rho : cp.Expression
        Matrix-valued CVXPY expression.
    dims : list[int]
        Dimensions of the tensor-product subsystems.
    axes : list[int]
        Subsystem indices to trace out.

    Returns
    -------
    cp.Expression
        CVXPY expression for the reduced operator.

    Notes
    -----
    The implementation uses the sparse superoperator returned by
    ``_partial_trace_superop`` and applies it to ``cp.vec(rho, order="F")``.
    """
    if not axes:
        return rho

    superop = _partial_trace_superop(dims, axes)
    out_dims = [dim for i, dim in enumerate(dims) if i not in axes]
    out_dim = int(np.prod(out_dims, dtype=int)) if out_dims else 1

    return cp.reshape(
        superop @ cp.vec(rho, order="F"),
        (out_dim, out_dim),
        order="F",
    )


def _ptrace_sum(rho: Any, dims: list[int], axes: list[int]):
    r"""
    Compute the partial trace of a sparse operator by the defining summation.

    Parameters
    ----------
    rho : Any
        Matrix-like input operator. The function first converts it to
        ``scipy.sparse.csr_array`` and then evaluates the defining summation.
    dims : list[int]
        Dimensions of the tensor-product subsystems.
    axes : list[int]
        Subsystem indices to trace out.

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
    if not axes:
        return _as_csr_array(rho)

    traced_rho = _as_csr_array(rho)
    current_dims = dims.copy()

    for axis in sorted(axes, reverse=True):
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


def ptrace(rho, dims, axes):
    """
    Compute the partial trace of an operator over selected subsystems.

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
    normalized_axes = _normalize_axes(axes, len(dims))

    if isinstance(rho, cp.Expression):
        return _ptrace_cvxpy(rho, dims, normalized_axes)
    if isinstance(rho, np.ndarray):
        return _ptrace_numpy(rho, dims, normalized_axes)
    return _ptrace_sum(rho, dims, normalized_axes)
