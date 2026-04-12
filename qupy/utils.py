from typing import Any

import numpy as np
import scipy.sparse as sp
import cvxpy as cp

def _identity_like(dim: int, template):
    """
    Return an identity matrix whose container type matches ``template``.

    Parameters
    ----------
    dim : int
        Dimension of the identity matrix.
    template : Any
        Reference object used to determine the output container type.

    Returns
    -------
    np.ndarray | sp.sparray
        Identity operator of size ``dim`` compatible with ``template``.

    Raises
    ------
    ValueError
        If ``dim`` is not positive.
    """
    if dim < 1:
        raise ValueError("identity dimension must be positive")

    if isinstance(template, cp.Expression):
        return sp.eye_array(dim)
    if sp.issparse(template):
        return sp.eye_array(dim)
    return np.eye(dim, dtype=np.result_type(template))

def _shape(op: np.ndarray | cp.Expression | sp.sparray) -> tuple[int, ...]:
    """
    Return the shape of an operator in a type-checker-friendly way.

    Parameters
    ----------
    op : np.ndarray | cp.Expression | sp.sparray
        Matrix-like object whose shape is requested.

    Returns
    -------
    tuple[int, ...]
        Shape tuple with entries converted to plain Python integers.
    """
    shape = getattr(op, "shape", None)
    if shape is None:
        shape = np.asarray(op).shape
    return tuple(int(dim) for dim in shape)

def _normalize_axes(axes: int | list[int], ndim: int) -> list[int]:
    """
    Normalize subsystem indices to a validated list of non-negative axes.

    Parameters
    ----------
    axes : int | list[int]
        Single axis or list of axes.
    ndim : int
        Number of available subsystems.

    Returns
    -------
    list[int]
        Normalized subsystem indices.

    Raises
    ------
    ValueError
        If an axis is out of range or appears more than once.
    """
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

def _as_csr_array(
    op: Any,
) -> sp.csr_array:
    """
    Convert a matrix-like object to ``scipy.sparse.csr_array``.

    Parameters
    ----------
    op : Any
        Matrix-like input object.

    Returns
    -------
    sp.csr_array
        CSR array representing ``op``.

    Raises
    ------
    TypeError
        If ``op`` is a CVXPY expression or cannot be interpreted as a matrix.
    """
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

def trace_product(A, B):
    """
    Compute the trace inner product ``Tr(A @ B)`` for mixed backends.

    Parameters
    ----------
    A, B : Any
        Matrix-like objects supported by NumPy, SciPy sparse, or CVXPY.

    Returns
    -------
    complex | np.number | cp.Expression
        Trace inner product of ``A`` and ``B``.
    """
    if isinstance(A, cp.Expression) or isinstance(B, cp.Expression):
        A_T = A.T if isinstance(A, cp.Expression) else np.asarray(A).T
        return cp.sum(cp.multiply(A_T, B))

    if sp.issparse(A) and sp.issparse(B):
        return A.multiply(B.T).sum()

    A_arr = A.toarray() if sp.issparse(A) else np.asarray(A)
    B_arr = B.toarray() if sp.issparse(B) else np.asarray(B)
    return np.sum(A_arr.T * B_arr)
