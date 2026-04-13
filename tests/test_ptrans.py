from collections.abc import Sequence

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from qupy import ptrans
from qupy.ptrans import _ptrans_cvxpy, _ptrans_numpy, _ptrans_sum
from qupy.utils import _normalize_axes
from tests.helpers import assert_allclose, random_matrix


def ref_ptrans(rho: np.ndarray, dims: Sequence[int], axes: int | Sequence[int]) -> np.ndarray:
    dims = list(dims)
    normalized_axes = _normalize_axes(axes, len(dims))
    tensor_rho = rho.reshape(dims + dims)
    perm = list(range(2 * len(dims)))
    for axis in normalized_axes:
        perm[axis], perm[axis + len(dims)] = perm[axis + len(dims)], perm[axis]
    dim = int(np.prod(dims, dtype=int)) if dims else 1
    return np.transpose(tensor_rho, axes=perm).reshape((dim, dim))


def test_ptrans_numpy_helper() -> None:
    rho = random_matrix((4, 4), seed=401, complex_=True)
    expected = ref_ptrans(rho, [2, 2], 1)
    assert_allclose(_ptrans_numpy(rho, [2, 2], 1), expected)


def test_ptrans_cvxpy_helper() -> None:
    rho = random_matrix((4, 4), seed=402, complex_=True)
    result = _ptrans_cvxpy(cp.Constant(rho), [2, 2], 1)
    assert result.value is not None
    assert_allclose(result.value, ref_ptrans(rho, [2, 2], 1))


def test_ptrans_sum_helper() -> None:
    rho = sp.csr_array(random_matrix((4, 4), seed=403, complex_=True))
    expected = ref_ptrans(rho.toarray(), [2, 2], 1)
    assert_allclose(_ptrans_sum(rho, [2, 2], 1), expected)


def test_ptrans_public_dense_sparse_and_cvxpy() -> None:
    rho = random_matrix((4, 4), seed=404, complex_=True)
    expected = ref_ptrans(rho, [2, 2], 1)

    assert_allclose(ptrans(rho, [2, 2], 1), expected)
    assert_allclose(ptrans(sp.csr_array(rho), [2, 2], -1), expected)

    expr = ptrans(cp.Constant(rho), [2, 2], 1)
    assert expr.value is not None
    assert_allclose(expr.value, expected)
