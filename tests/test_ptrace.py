from collections.abc import Sequence

import cvxpy as cp
import numpy as np
import pytest
import scipy.sparse as sp

from qitrix import ptrace
from qitrix.ops.ptrace import (
    _partial_trace_superop,
    _ptrace_cvxpy,
    _ptrace_numpy,
    _ptrace_sum,
)
from qitrix._internal.utils import normalize_axes
from tests.helpers import assert_allclose, random_matrix


def ref_ptrace(rho: np.ndarray, dims: Sequence[int], axes: int | Sequence[int]) -> np.ndarray:
    dims = list(dims)
    normalized_axes = normalize_axes(axes, len(dims))
    tensor_rho = rho.reshape(dims + dims)
    current_dims = dims.copy()

    for axis in sorted(normalized_axes, reverse=True):
        tensor_rho = np.trace(
            tensor_rho,
            axis1=axis,
            axis2=axis + len(current_dims),
        )
        del current_dims[axis]

    out_dim = int(np.prod(current_dims, dtype=int)) if current_dims else 1
    return tensor_rho.reshape((out_dim, out_dim))


def test_ptrace_numpy_helper() -> None:
    rho = random_matrix((6, 6), seed=301, complex_=True)
    expected = ref_ptrace(rho, [2, 3], 0)
    assert_allclose(_ptrace_numpy(rho, [2, 3], 0), expected)


def test_partial_trace_superop_matches_reference() -> None:
    dims = [2, 2]
    rho = random_matrix((4, 4), seed=302, complex_=True)
    superop = _partial_trace_superop(dims, 1)
    vec_rho = rho.reshape((-1, 1), order="F")
    expected = ref_ptrace(rho, dims, 1).reshape((-1, 1), order="F")
    assert_allclose(superop @ vec_rho, expected)


def test_ptrace_cvxpy_helper() -> None:
    rho = random_matrix((4, 4), seed=303, complex_=True)
    result = _ptrace_cvxpy(cp.Constant(rho), [2, 2], 1)
    assert result.value is not None
    assert_allclose(result.value, ref_ptrace(rho, [2, 2], 1))


def test_ptrace_sum_helper() -> None:
    rho = sp.csr_array(random_matrix((4, 4), seed=304, complex_=True))
    expected = ref_ptrace(rho.toarray(), [2, 2], 1)
    assert_allclose(_ptrace_sum(rho, [2, 2], 1), expected)


def test_ptrace_public_dense_sparse_and_cvxpy() -> None:
    rho = random_matrix((4, 4), seed=305, complex_=True)
    expected = ref_ptrace(rho, [2, 2], 1)

    assert_allclose(ptrace(rho, [2, 2], 1), expected)
    assert_allclose(ptrace(sp.csr_array(rho), [2, 2], -1), expected)

    expr = ptrace(cp.Constant(rho), [2, 2], 1)
    assert expr.value is not None
    assert_allclose(expr.value, expected)


def test_ptrace_rejects_dimension_mismatch() -> None:
    rho = np.eye(4)
    with pytest.raises(ValueError, match="expected a square matrix of shape \\(6, 6\\)"):
        ptrace(rho, [2, 3], 0)


def test_ptrace_rejects_nonsquare_input() -> None:
    rho = np.ones((4, 2))
    with pytest.raises(ValueError, match="expected a square matrix"):
        ptrace(rho, [2, 2], 0)


def test_ptrace_preserves_trace() -> None:
    rho = random_matrix((12, 12), seed=306, complex_=True)
    reduced = ptrace(rho, [2, 2, 3], axes=[0, 2])
    assert np.isclose(np.trace(reduced), np.trace(rho))
