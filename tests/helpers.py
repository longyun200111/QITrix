import numpy as np
import scipy.sparse as sp

ATOL = 1e-10


def to_dense(op: np.ndarray | sp.spmatrix | sp.sparray) -> np.ndarray:
    if sp.issparse(op):
        return op.toarray()
    return np.asarray(op)


def assert_allclose(
    actual: np.ndarray | sp.spmatrix | sp.sparray,
    expected: np.ndarray,
) -> None:
    actual_dense = to_dense(actual)
    assert np.allclose(actual_dense, expected, atol=ATOL, rtol=0.0)


def random_matrix(
    shape: tuple[int, int],
    seed: int,
    complex_: bool = False,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    real = rng.standard_normal(shape)
    if complex_:
        imag = rng.standard_normal(shape)
        return real + 1j * imag
    return real
