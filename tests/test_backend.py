import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from QITrix.backend import backend_registry, resolve_backend
from QITrix.ops.extend import extend
from QITrix.ops.tensor import tensor
from tests.helpers import assert_allclose, random_matrix


def test_backend_registry_contains_builtin_adapters() -> None:
    kinds = [adapter.kind for adapter in backend_registry.adapters]
    assert kinds == ["symbolic", "sparse", "dense"]


def test_resolve_backend_priority() -> None:
    dense = np.eye(2)
    sparse = sp.csr_array(np.eye(2))
    expr = cp.Constant(np.eye(2))

    assert resolve_backend(dense).kind == "dense"
    assert resolve_backend(dense, sparse).kind == "sparse"
    assert resolve_backend(dense, sparse, expr).kind == "symbolic"


def test_tensor_and_extend_keep_sparse_outputs() -> None:
    op = sp.csr_array(random_matrix((2, 2), seed=601))
    tensor_result = tensor(op, np.eye(2))
    extend_result = extend(op, dims=[2, 2], axes=0)

    assert sp.issparse(tensor_result)
    assert sp.issparse(extend_result)
    assert_allclose(tensor_result, np.kron(op.toarray(), np.eye(2)))
    assert_allclose(extend_result, np.kron(op.toarray(), np.eye(2)))
