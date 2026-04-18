import numpy as np
import pytest
import scipy.sparse as sp

from qitrix import fock, fock_dm
from tests.helpers import assert_allclose


@pytest.mark.parametrize(
    ("fmt", "expected_type"),
    [
        ("dense", np.ndarray),
        ("csr", sp.csr_array),
        ("coo", sp.coo_array),
    ],
)
def test_fock_returns_expected_formats(fmt: str, expected_type: type) -> None:
    result = fock(4, 2, format=fmt)
    assert isinstance(result, expected_type)
    assert_allclose(result, np.array([[0.0], [0.0], [1.0], [0.0]]))


@pytest.mark.parametrize(
    ("fmt", "expected_type"),
    [
        ("dense", np.ndarray),
        ("csr", sp.csr_array),
        ("coo", sp.coo_array),
    ],
)
def test_fock_dm_returns_expected_formats(fmt: str, expected_type: type) -> None:
    result = fock_dm(4, 2, format=fmt)
    assert isinstance(result, expected_type)
    assert_allclose(result, np.diag([0.0, 0.0, 1.0, 0.0]))


@pytest.mark.parametrize("fn", [fock, fock_dm])
def test_fock_variants_reject_nonpositive_dimension(fn) -> None:
    with pytest.raises(ValueError, match="dim must be positive"):
        fn(0, 0)


@pytest.mark.parametrize("fn", [fock, fock_dm])
def test_fock_variants_reject_invalid_occupation_number(fn) -> None:
    with pytest.raises(ValueError, match="0 <= n < dim"):
        fn(4, -1)

    with pytest.raises(ValueError, match="0 <= n < dim"):
        fn(4, 4)
