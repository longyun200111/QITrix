import numpy as np
import pytest
import scipy.sparse as sp

from qupy import fock, fock_dm
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
