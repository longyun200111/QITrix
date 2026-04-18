import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from qitrix import tensor
from qitrix.ops.tensor import _kron
from tests.helpers import assert_allclose, random_matrix


def test_kron_dense() -> None:
    a = random_matrix((2, 2), seed=501)
    b = random_matrix((2, 2), seed=502)
    result = _kron(a, b)
    assert isinstance(result, np.ndarray)
    assert_allclose(result, np.kron(a, b))


def test_kron_sparse() -> None:
    a = sp.csr_array(random_matrix((2, 2), seed=503))
    b = sp.coo_array(random_matrix((2, 2), seed=504))
    result = _kron(a, b)
    assert sp.issparse(result)
    assert_allclose(result, np.kron(a.toarray(), b.toarray()))


def test_kron_cvxpy() -> None:
    a = cp.Constant(random_matrix((2, 2), seed=505))
    b = random_matrix((2, 2), seed=506)
    result = _kron(a, b)
    assert isinstance(result, cp.Expression)
    assert result.value is not None
    assert_allclose(result.value, np.kron(a.value, b))


def test_tensor_dense_sparse_and_cvxpy() -> None:
    a = random_matrix((2, 2), seed=507)
    b = random_matrix((2, 2), seed=508)
    c = random_matrix((1, 1), seed=509)
    expected = np.kron(np.kron(a, b), c)

    assert_allclose(tensor(a, b, c), expected)
    assert_allclose(tensor(sp.csr_array(a), sp.coo_array(b), c), expected)

    expr = tensor(cp.Constant(a), b, c)
    assert isinstance(expr, cp.Expression)
    assert expr.value is not None
    assert_allclose(expr.value, expected)
