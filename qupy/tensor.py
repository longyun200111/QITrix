import cvxpy as cp
import numpy as np
import scipy.sparse as sp


def _kron(left, right):
    """
    Compute a Kronecker product while preserving the most expressive type.

    Parameters
    ----------
    left, right : Any
        Matrix-like operands. Dense NumPy arrays, SciPy sparse matrices/arrays,
        and CVXPY expressions are supported.

    Returns
    -------
    np.ndarray | sp.sparray | cp.Expression
        Kronecker product of ``left`` and ``right``.

    Notes
    -----
    If either operand is a CVXPY expression, the result is built with
    ``cp.kron``. Otherwise, sparse operands are handled with ``sp.kron`` and
    dense operands with ``np.kron``.
    """
    if isinstance(left, cp.Expression) or isinstance(right, cp.Expression):
        return cp.kron(left, right)
    if sp.issparse(left) or sp.issparse(right):
        return sp.kron(left, right)
    return np.kron(np.asarray(left), np.asarray(right))


def tensor(*factors):
    """
    Compute the tensor product of multiple factors from left to right.

    Parameters
    ----------
    *factors : tuple[Any, ...]
        Sequence of matrix-like factors to be tensored together.

    Returns
    -------
    np.ndarray | sp.sparray | cp.Expression
        Tensor product of all supplied factors.

    Raises
    ------
    ValueError
        If no factor is provided.
    """
    if not factors:
        raise ValueError("at least one factor is required")
    result = factors[0]
    for factor in factors[1:]:
        result = _kron(result, factor)
    return result
