from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


def normalize_labels(labels: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(labels, str):
        return (labels,)
    return tuple(str(label) for label in labels)


@dataclass(frozen=True, slots=True)
class Space:
    label: str
    dim: int

    def __init__(self, label: str, dim: int):
        normalized_label = str(label)
        normalized_dim = int(dim)
        if normalized_dim < 1:
            raise ValueError("space dimension must be a positive integer")
        object.__setattr__(self, "label", normalized_label)
        object.__setattr__(self, "dim", normalized_dim)

    @property
    def labels(self) -> tuple[str]:
        return (self.label,)

    @property
    def dims(self) -> tuple[int]:
        return (self.dim,)

    def tensor(self, other: Space | SpaceList) -> SpaceList:
        return SpaceList([self]).tensor(other)


@dataclass(frozen=True, slots=True)
class SpaceList:
    spaces: tuple[Space, ...]

    def __init__(self, spaces: Iterable[Space]):
        normalized = tuple(_coerce_factor_space(space) for space in spaces)
        labels = tuple(space.label for space in normalized)
        if len(set(labels)) != len(labels):
            raise ValueError("space labels must be unique")
        object.__setattr__(self, "spaces", normalized)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(space.label for space in self.spaces)

    @property
    def dims(self) -> tuple[int, ...]:
        return tuple(space.dim for space in self.spaces)

    @property
    def dim(self) -> int:
        return int(np.prod(self.dims, dtype=int)) if self.spaces else 1

    @property
    def subsystems(self) -> tuple[tuple[str, int], ...]:
        return tuple((space.label, space.dim) for space in self.spaces)

    def index(self, label: str) -> int:
        try:
            return self.labels.index(label)
        except ValueError as exc:
            raise ValueError(f"unknown subsystem label {label!r}") from exc

    def indices(self, labels: str | Sequence[str]) -> list[int]:
        normalized = normalize_labels(labels)
        if len(set(normalized)) != len(normalized):
            raise ValueError("subsystem labels must be unique")
        return [self.index(label) for label in normalized]

    def perm(self, labels: Sequence[str]) -> list[int]:
        normalized = tuple(str(label) for label in labels)
        if set(normalized) != set(self.labels) or len(normalized) != len(self.labels):
            raise ValueError("labels must be a permutation of the space labels")
        return [self.index(label) for label in normalized]

    def select(self, labels: str | Sequence[str]) -> SpaceList:
        normalized = normalize_labels(labels)
        indices = self.indices(normalized)
        return SpaceList(self.spaces[index] for index in indices)

    def complement(self, labels: str | Sequence[str]) -> SpaceList:
        normalized = set(normalize_labels(labels))
        missing = [space for space in self.spaces if space.label not in normalized]
        return SpaceList(missing)

    def reordered(self, labels: Sequence[str]) -> SpaceList:
        return self.select(labels)

    def drop(self, labels: str | Sequence[str]) -> SpaceList:
        return self.complement(labels)

    def tensor(self, other: Space | SpaceList) -> SpaceList:
        right = as_space_list(other)
        overlap = set(self.labels) & set(right.labels)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"cannot tensor spaces with repeated labels: {names}")
        return SpaceList(self.spaces + right.spaces)


def _coerce_factor_space(space: Space) -> Space:
    if not isinstance(space, Space):
        raise TypeError("SpaceList expects Space factors")
    return space


def as_space_list(space: Space | SpaceList) -> SpaceList:
    if isinstance(space, SpaceList):
        return space
    if isinstance(space, Space):
        return SpaceList([space])
    raise TypeError("expected a Space or SpaceList")
