import cvxpy as cp
import numpy as np
import pytest
import scipy.sparse as sp

from QITrix import Operator, Space, SpaceList, operator, permute, ptrace, ptrans, tensor
from tests.helpers import assert_allclose, random_matrix


def test_space_and_space_list_validation_and_queries() -> None:
    factor = Space("A", 2)
    space = SpaceList([Space("A", 2), Space("B", 3), Space("C", 2)])
    assert factor.label == "A"
    assert factor.dim == 2
    assert space.labels == ("A", "B", "C")
    assert space.dims == (2, 3, 2)
    assert space.dim == 12
    assert space.index("B") == 1
    assert space.indices(["C", "A"]) == [2, 0]
    assert space.reordered(["C", "A", "B"]).labels == ("C", "A", "B")
    assert space.drop("B").labels == ("A", "C")


def test_space_rejects_invalid_labels_and_dims() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        Space("A", 0)
    with pytest.raises(ValueError, match="unique"):
        SpaceList([Space("A", 2), Space("A", 3)])
    with pytest.raises(ValueError, match="unknown subsystem label"):
        SpaceList([Space("A", 2)]).index("B")
    with pytest.raises(TypeError, match="Space or SpaceList"):
        Operator(np.eye(2), space=[("A", 2)])  # type: ignore[arg-type]


def test_operator_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="incompatible with spaces"):
        Operator(np.eye(4), space=SpaceList([Space("A", 2), Space("B", 3)]))


def test_operator_ptrace_ptrans_and_permute_match_raw_functions() -> None:
    space = SpaceList([Space("A", 2), Space("B", 2)])
    rho = random_matrix((4, 4), seed=701, complex_=True)
    op = Operator(rho, space=space)

    traced = ptrace(op, "B")
    transposed = ptrans(op, "B")
    permuted = permute(op, ["B", "A"])

    assert traced.space.labels == ("A",)
    assert permuted.input_space.labels == ("B", "A")
    assert permuted.output_space.labels == ("B", "A")
    assert_allclose(traced.as_raw(), ptrace(rho, [2, 2], 1))
    assert_allclose(transposed.as_raw(), ptrans(rho, [2, 2], 1))
    assert_allclose(permuted.as_raw(), permute(rho, [2, 2], [1, 0]))


def test_operator_permute_updates_only_selected_side() -> None:
    op = Operator(
        np.arange(24).reshape(4, 6),
        input_space=SpaceList([Space("A", 2), Space("B", 3)]),
        output_space=SpaceList([Space("X", 2), Space("Y", 2)]),
    )

    left = op.permute(["Y", "X"], direction="left")
    right = op.permute(["B", "A"], direction="right")

    assert left.output_space.labels == ("Y", "X")
    assert left.input_space.labels == ("A", "B")
    assert right.input_space.labels == ("B", "A")
    assert right.output_space.labels == ("X", "Y")
    assert_allclose(left.as_raw(), permute(op.as_raw(), [2, 2], [1, 0], direction="left"))
    assert_allclose(right.as_raw(), permute(op.as_raw(), [2, 3], [1, 0], direction="right"))


def test_operator_extend_rectangular_operator() -> None:
    base = Operator(
        np.array([[1.0, 0.0]], dtype=complex),
        input_space=Space("A", 2),
        output_space=Space("A", 1),
    )
    extended = base.extend(
        input_space=SpaceList([Space("A", 2), Space("B", 2)]),
        output_space=SpaceList([Space("A", 1), Space("B", 2)]),
        input_labels="A",
        output_labels="A",
    )

    expected = np.kron(base.as_raw(), np.eye(2))
    assert extended.input_space.labels == ("A", "B")
    assert extended.output_space.labels == ("A", "B")
    assert_allclose(extended.as_raw(), expected)


def test_operator_tensor_preserves_spaces_and_supports_backends() -> None:
    left = Operator(np.eye(2), space=Space("A", 2))
    right = Operator(sp.csr_array(np.eye(3)), space=Space("B", 3))
    tensored = left.tensor(right)

    assert tensored.space.labels == ("A", "B")
    assert sp.issparse(tensored.as_raw())
    assert_allclose(tensored.as_raw(), np.eye(6))

    symbolic = Operator(cp.Constant(np.eye(2)), space=Space("C", 2))
    symbolic_tensor = left.tensor(symbolic)
    assert isinstance(symbolic_tensor.as_raw(), cp.Expression)
    assert symbolic_tensor.as_raw().value is not None
    assert_allclose(symbolic_tensor.as_raw().value, np.eye(4))


def test_operator_helper_constructor() -> None:
    op = operator(np.eye(2), space=Space("A", 2))
    assert isinstance(op, Operator)
    assert op.space.labels == ("A",)


def test_operator_extend_normalizes_single_space_targets() -> None:
    base = Operator(np.eye(2), space=Space("A", 2))
    extended = base.extend(space=SpaceList([Space("A", 2), Space("B", 2)]), labels="A")
    assert extended.space.labels == ("A", "B")
    assert_allclose(extended.as_raw(), np.kron(np.eye(2), np.eye(2)))
