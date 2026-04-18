import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from QITrix import extend
from QITrix.ops.extend import _inverse_axis_order
from tests.helpers import assert_allclose, random_matrix


def test_inverse_axis_order() -> None:
    assert _inverse_axis_order([2, 0, 1]) == [1, 2, 0]


def test_extend_public_dense_sparse_and_cvxpy() -> None:
    x = random_matrix((2, 2), seed=101)
    expected_left = np.kron(x, np.eye(3))
    expected_right = np.kron(np.eye(3), x)

    assert_allclose(extend(x, dims=[2, 3], axes=0), expected_left)
    assert_allclose(extend(sp.csr_array(x), dims=[3, 2], axes=1), expected_right)

    expr = extend(cp.Constant(x), dims=[2, 3], axes=0)
    assert isinstance(expr, cp.Expression)
    assert expr.value is not None
    assert_allclose(expr.value, expected_left)


def test_extend_public_rectangular_operator() -> None:
    op = random_matrix((1, 2), seed=102)
    expected = np.kron(op, np.eye(2))
    result = extend(
        op,
        input_dims=[2, 2],
        output_dims=[1, 2],
        input_axes=0,
        output_axes=0,
    )
    assert_allclose(result, expected)
