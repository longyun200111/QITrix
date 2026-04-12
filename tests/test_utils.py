import cvxpy as cp
import numpy as np
import pytest
import scipy.sparse as sp

from qupy.utils import (
    _as_csr_array,
    _as_sparse_array,
    _identity_like,
    _normalize_axes,
    _shape,
    trace_product,
)
from tests.helpers import assert_allclose, random_matrix


def test_as_sparse_array_from_dense() -> None:
    dense = random_matrix((2, 2), seed=601)
    converted = _as_sparse_array(dense)
    assert isinstance(converted, sp.coo_array)
    assert_allclose(converted, dense)


def test_as_sparse_array_from_sparse_matrix() -> None:
    matrix = sp.csr_matrix(random_matrix((2, 2), seed=602))
    converted = _as_sparse_array(matrix)
    assert isinstance(converted, sp.csr_array)
    assert_allclose(converted, matrix.toarray())


def test_identity_like_matches_backend() -> None:
    dense_identity = _identity_like(3, random_matrix((1, 1), seed=603))
    sparse_identity = _identity_like(3, sp.csr_array(random_matrix((1, 1), seed=604)))
    expr_identity = _identity_like(3, cp.Constant(random_matrix((1, 1), seed=605)))

    assert isinstance(dense_identity, np.ndarray)
    assert sp.issparse(sparse_identity)
    assert sp.issparse(expr_identity)
    assert_allclose(dense_identity, np.eye(3))
    assert_allclose(sparse_identity, np.eye(3))
    assert_allclose(expr_identity, np.eye(3))


def test_shape_supports_dense_sparse_and_cvxpy() -> None:
    assert _shape(np.zeros((2, 3))) == (2, 3)
    assert _shape(sp.csr_array((2, 3))) == (2, 3)
    assert _shape(cp.Constant(np.zeros((2, 3)))) == (2, 3)


def test_normalize_axes_supports_int_and_negative_indices() -> None:
    assert _normalize_axes(1, 3) == [1]
    assert _normalize_axes([-1, 0], 3) == [2, 0]


def test_normalize_axes_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="axes must be unique"):
        _normalize_axes([0, 0], 2)


def test_as_csr_array_supports_dense_and_sparse() -> None:
    dense_input = random_matrix((2, 2), seed=606)
    sparse_input = random_matrix((2, 2), seed=607)
    dense = _as_csr_array(dense_input)
    sparse = _as_csr_array(sp.coo_array(sparse_input))

    assert isinstance(dense, sp.csr_array)
    assert isinstance(sparse, sp.csr_array)
    assert_allclose(dense, dense_input)
    assert_allclose(sparse, sparse_input)


def test_as_csr_array_rejects_cvxpy() -> None:
    with pytest.raises(TypeError, match="CVXPY expression"):
        _as_csr_array(cp.Constant(np.eye(2)))


def test_trace_product_dense_sparse_and_cvxpy() -> None:
    a = random_matrix((2, 2), seed=608)
    b = random_matrix((2, 2), seed=609)
    expected = np.trace(a @ b)

    assert np.isclose(trace_product(a, b), expected)
    assert np.isclose(trace_product(sp.csr_array(a), sp.coo_array(b)), expected)

    expr_value = trace_product(cp.Constant(a), b).value
    assert expr_value is not None
    assert np.isclose(expr_value, expected)
