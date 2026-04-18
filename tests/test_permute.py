from collections.abc import Sequence

import cvxpy as cp
import numpy as np
import pytest
import scipy.sparse as sp

from QITrix import permute
from QITrix.ops.permute import _get_permutation_matrix, _permute_numpy
from tests.helpers import assert_allclose, random_matrix


def ref_permute(
    rho: np.ndarray,
    dims: Sequence[int],
    perm: Sequence[int],
    direction: str = "both",
) -> np.ndarray:
    dims = list(dims)
    perm = list(perm)
    dim = int(np.prod(dims, dtype=int)) if dims else 1
    if direction == "both":
        tensor_rho = rho.reshape(dims + dims)
        tensor_perm = perm + [p + len(dims) for p in perm]
        return np.transpose(tensor_rho, axes=tensor_perm).reshape((dim, dim))
    if direction == "left":
        tensor_rho = rho.reshape((*dims, -1))
        tensor_perm = perm + [len(dims)]
        return np.transpose(tensor_rho, axes=tensor_perm).reshape((dim, -1))
    if direction == "right":
        tensor_rho = rho.reshape((-1, *dims))
        tensor_perm = [0] + [p + 1 for p in perm]
        return np.transpose(tensor_rho, axes=tensor_perm).reshape((-1, dim))
    raise ValueError("invalid direction")


@pytest.mark.parametrize("direction", ["both", "left", "right"])
def test_permute_numpy_helper(direction: str) -> None:
    dims = [2, 3]
    rho = random_matrix((6, 6), seed=201)
    assert_allclose(
        _permute_numpy(rho, dims, [1, 0], direction),
        ref_permute(rho, dims, [1, 0], direction),
    )


def test_get_permutation_matrix() -> None:
    dims = [2, 3]
    perm_matrix = _get_permutation_matrix(dims, [1, 0])
    basis = random_matrix((6, 1), seed=202).reshape(-1)
    reshaped = basis.reshape(dims)
    expected = np.transpose(reshaped, [1, 0]).reshape(-1)
    assert_allclose(perm_matrix @ basis[:, None], expected[:, None])


@pytest.mark.parametrize("direction", ["both", "left", "right"])
def test_permute_public_all_supported_types(direction: str) -> None:
    dims = [2, 3]
    rho = random_matrix((6, 6), seed=203, complex_=True)
    expected = ref_permute(rho, dims, [1, 0], direction)

    assert_allclose(permute(rho, dims, [1, 0], direction), expected)
    assert_allclose(permute(sp.csr_array(rho), dims, [1, 0], direction), expected)

    expr = permute(cp.Constant(rho), dims, [1, 0], direction)
    assert isinstance(expr, cp.Expression)
    assert expr.value is not None
    assert_allclose(expr.value, expected)


@pytest.mark.parametrize(
    ("dims", "perm", "match"),
    [
        ([2, 2], [0, 0], "perm must be a permutation"),
        ([2, 2], [0], "perm must be a permutation"),
        ([2, 0], [1, 0], "dims must contain only positive integers"),
    ],
)
def test_permute_rejects_invalid_dims_and_permutations(
    dims: list[int], perm: list[int], match: str
) -> None:
    rho = np.eye(4)
    with pytest.raises(ValueError, match=match):
        permute(rho, dims, perm)


def test_permute_rejects_shape_mismatch() -> None:
    rho = np.eye(4)
    with pytest.raises(ValueError, match="expected shape \\(6, 6\\)"):
        permute(rho, [2, 3], [1, 0])


def test_permute_then_inverse_recovers_operator() -> None:
    dims = [2, 3, 2]
    perm = [2, 0, 1]
    inverse_perm = [1, 2, 0]
    rho = random_matrix((12, 12), seed=204, complex_=True)

    permuted = permute(rho, dims, perm)
    recovered = permute(permuted, [dims[i] for i in perm], inverse_perm)

    assert_allclose(recovered, rho)
