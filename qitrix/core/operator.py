from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from qitrix._internal.utils import shape_of

from .space import Space, SpaceList, as_space_list, normalize_labels


@dataclass(frozen=True, slots=True)
class Operator:
    data: Any
    input_space: SpaceList
    output_space: SpaceList

    def __init__(
        self,
        data: Any,
        space: Space | SpaceList | None = None,
        *,
        input_space: Space | SpaceList | None = None,
        output_space: Space | SpaceList | None = None,
    ):
        if space is not None:
            if input_space is not None or output_space is not None:
                raise ValueError("provide either space or input_space/output_space, not both")
            input_space = space
            output_space = space
        if input_space is None or output_space is None:
            raise ValueError("space or both input_space/output_space must be provided")
        input_space = as_space_list(input_space)
        output_space = as_space_list(output_space)

        expected_shape = (output_space.dim, input_space.dim)
        if shape_of(data) != expected_shape:
            raise ValueError(
                f"operator shape {shape_of(data)} is incompatible with spaces of shape {expected_shape}"
            )

        object.__setattr__(self, "data", data)
        object.__setattr__(self, "input_space", input_space)
        object.__setattr__(self, "output_space", output_space)

    @property
    def space(self) -> SpaceList:
        if self.input_space != self.output_space:
            raise ValueError("space is only defined for square operators")
        return self.input_space

    def as_raw(self) -> Any:
        return self.data

    def with_spaces(
        self,
        *,
        input_space: Space | SpaceList | None = None,
        output_space: Space | SpaceList | None = None,
    ) -> Operator:
        return Operator(
            self.data,
            input_space=input_space or self.input_space,
            output_space=output_space or self.output_space,
        )

    def relabel(
        self,
        *,
        input_space: Space | SpaceList | None = None,
        output_space: Space | SpaceList | None = None,
    ) -> Operator:
        return self.with_spaces(input_space=input_space, output_space=output_space)

    def tensor(self, other: Operator) -> Operator:
        from qitrix.ops.tensor import tensor

        return Operator(
            tensor(self.data, other.data),
            input_space=self.input_space.tensor(other.input_space),
            output_space=self.output_space.tensor(other.output_space),
        )

    def ptrace(self, labels: str | Sequence[str]) -> Operator:
        from qitrix.ops.ptrace import ptrace

        return ptrace(self, labels)

    def ptrans(self, labels: str | Sequence[str]) -> Operator:
        from qitrix.ops.ptrans import ptrans

        return ptrans(self, labels)

    def permute(self, labels: Sequence[str], direction: str = "both") -> Operator:
        from qitrix.ops.permute import permute

        return permute(self, labels, direction=direction)

    def extend(
        self,
        *,
        space: Space | SpaceList | None = None,
        labels: str | Sequence[str] | None = None,
        input_space: Space | SpaceList | None = None,
        output_space: Space | SpaceList | None = None,
        input_labels: str | Sequence[str] | None = None,
        output_labels: str | Sequence[str] | None = None,
    ) -> Operator:
        from qitrix.ops.extend import extend

        if space is not None:
            if input_space is not None or output_space is not None:
                raise ValueError("provide either space or input_space/output_space, not both")
            input_space = space
            output_space = space
        if labels is not None:
            if input_labels is not None or output_labels is not None:
                raise ValueError("provide either labels or input_labels/output_labels, not both")
            input_labels = labels
            output_labels = labels
        if input_space is None or output_space is None:
            raise ValueError("target input_space/output_space must be provided")
        if input_labels is None or output_labels is None:
            raise ValueError("target input_labels/output_labels must be provided")
        input_space = as_space_list(input_space)
        output_space = as_space_list(output_space)

        input_axes = input_space.indices(normalize_labels(input_labels))
        output_axes = output_space.indices(normalize_labels(output_labels))

        data = extend(
            self.data,
            input_dims=input_space.dims,
            output_dims=output_space.dims,
            input_axes=input_axes,
            output_axes=output_axes,
        )
        return Operator(data, input_space=input_space, output_space=output_space)


def operator(
    data: Any,
    space: Space | SpaceList | None = None,
    *,
    input_space: Space | SpaceList | None = None,
    output_space: Space | SpaceList | None = None,
) -> Operator:
    return Operator(data, space, input_space=input_space, output_space=output_space)
