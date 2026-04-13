from collections.abc import Sequence
from typing import Any, overload

import numpy as np

from .permute import permute
from .tensor import tensor
from ._types import Expression, NDArray, OperatorLike, SparseArray, SparseLike
from .utils import _normalize_axes, _shape, _identity_like, _as_sparse_array


def _inverse_axis_order(axis_order: Sequence[int]) -> list[int]:
    """
    Compute the inverse permutation that restores canonical subsystem order.

    Parameters
    ----------
    axis_order : Sequence[int]
        Current ordering of subsystem labels.

    Returns
    -------
    list[int]
        Permutation that maps ``axis_order`` back to
        ``[0, 1, ..., len(axis_order) - 1]``.
    """
    inverse = [-1] * len(axis_order)
    for current_position, axis_label in enumerate(axis_order):
        inverse[axis_label] = current_position
    return inverse


@overload
def extend(
    op: Expression,
    dims: Sequence[int] | None = None,
    axes: int | Sequence[int] | None = None,
    input_dims: Sequence[int] | None = None,
    output_dims: Sequence[int] | None = None,
    input_axes: int | Sequence[int] | None = None,
    output_axes: int | Sequence[int] | None = None,
) -> Expression: ...


@overload
def extend(
    op: SparseLike,
    dims: Sequence[int] | None = None,
    axes: int | Sequence[int] | None = None,
    input_dims: Sequence[int] | None = None,
    output_dims: Sequence[int] | None = None,
    input_axes: int | Sequence[int] | None = None,
    output_axes: int | Sequence[int] | None = None,
) -> SparseArray: ...


@overload
def extend(
    op: NDArray[Any],
    dims: Sequence[int] | None = None,
    axes: int | Sequence[int] | None = None,
    input_dims: Sequence[int] | None = None,
    output_dims: Sequence[int] | None = None,
    input_axes: int | Sequence[int] | None = None,
    output_axes: int | Sequence[int] | None = None,
) -> NDArray[Any]: ...


def extend(
    op: OperatorLike,
    dims: Sequence[int] | None = None,
    axes: Sequence[int] | int | None = None,
    input_dims: Sequence[int] | None = None,
    output_dims: Sequence[int] | None = None,
    input_axes: Sequence[int] | int | None = None,
    output_axes: Sequence[int] | int | None = None,
) -> OperatorLike:
    """
    Extend an operator to a larger multipartite system.

    Parameters
    ----------
    op : np.ndarray | scipy.sparse.spmatrix | sp.sparray | cvxpy.Expression
        Operator acting on a subsystem selection.
    dims : Sequence[int] | None, optional
        Common input/output subsystem dimensions. This is a shorthand for
        setting both ``input_dims`` and ``output_dims``.
    axes : int | Sequence[int] | None, optional
        Common input/output subsystem indices. This is a shorthand for setting
        both ``input_axes`` and ``output_axes``.
    input_dims, output_dims : Sequence[int] | None, optional
        Input and output subsystem dimensions for a rectangular operator.
    input_axes, output_axes : int | Sequence[int] | None, optional
        Positions of the subsystems on which ``op`` acts in the input and
        output spaces, respectively.

    Returns
    -------
    np.ndarray | cvxpy.Expression | sp.sparray
        Extended operator acting on the full multipartite space.

    Notes
    -----
    The function tensors ``op`` with an identity on untouched subsystems and
    then permutes the resulting factors back to the canonical subsystem order.
    """
    if isinstance(op, SparseLike):
        op = _as_sparse_array(op)

    input_dims = dims if input_dims is None else input_dims
    output_dims = dims if output_dims is None else output_dims
    input_axes = axes if input_axes is None else input_axes
    output_axes = axes if output_axes is None else output_axes

    if input_dims is None or output_dims is None:
        raise ValueError("dims or both input_dims/output_dims must be provided")
    if input_axes is None or output_axes is None:
        raise ValueError("axes or both input_axes/output_axes must be provided")

    input_axes = _normalize_axes(input_axes, len(input_dims))
    output_axes = _normalize_axes(output_axes, len(output_dims))

    remaining_input_axes = [i for i in range(len(input_dims)) if i not in input_axes]
    remaining_output_axes = [i for i in range(len(output_dims)) if i not in output_axes]

    remaining_input_dims = [input_dims[i] for i in remaining_input_axes]
    remaining_output_dims = [output_dims[i] for i in remaining_output_axes]

    if remaining_input_dims != remaining_output_dims:
        raise ValueError(
            "untouched input/output subsystem dimensions must match so that the "
            "extension can tensor an identity operator on them"
        )

    selected_input_dim = int(np.prod([input_dims[i] for i in input_axes], dtype=int))
    selected_output_dim = int(np.prod([output_dims[i] for i in output_axes], dtype=int))
    op_shape = _shape(op)

    if op_shape != (selected_output_dim, selected_input_dim):
        raise ValueError(
            "op has incompatible shape: expected "
            f"{(selected_output_dim, selected_input_dim)}, got {op_shape}"
        )

    remaining_dim = int(np.prod(remaining_input_dims, dtype=int)) if remaining_input_dims else 1
    identity = _identity_like(remaining_dim, op)
    ext_op = tensor(op, identity)

    base_output_axes = output_axes + remaining_output_axes
    base_input_axes = input_axes + remaining_input_axes

    base_output_dims = [output_dims[i] for i in base_output_axes]
    base_input_dims = [input_dims[i] for i in base_input_axes]

    perm_out = _inverse_axis_order(base_output_axes)
    perm_in = _inverse_axis_order(base_input_axes)

    ext_op = permute(ext_op, base_output_dims, perm_out, direction="left")
    ext_op = permute(ext_op, base_input_dims, perm_in, direction="right")

    return ext_op
