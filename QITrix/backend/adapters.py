from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import cvxpy as cp
import numpy as np
import scipy.sparse as sp


def to_sparse_array(op: Any) -> sp.sparray:
    if sp.issparse(op):
        if isinstance(op, sp.sparray):
            return op
        if isinstance(op, sp.csr_matrix):
            return sp.csr_array(op)
        if isinstance(op, sp.csc_matrix):
            return sp.csc_array(op)
        if isinstance(op, sp.coo_matrix):
            return sp.coo_array(op)
        if isinstance(op, sp.bsr_matrix):
            return sp.bsr_array(op)
        if isinstance(op, sp.dia_matrix):
            return sp.dia_array(op)
        if isinstance(op, sp.dok_matrix):
            return sp.dok_array(op)
        if isinstance(op, sp.lil_matrix):
            return sp.lil_array(op)
    try:
        return sp.coo_array(op)
    except Exception as exc:
        raise ValueError("input cannot be converted to a sparse array") from exc


class BackendAdapter(ABC):
    kind: str
    priority: int

    @abstractmethod
    def can_handle(self, obj: Any) -> bool: ...

    @abstractmethod
    def asarray(self, obj: Any) -> Any: ...

    @abstractmethod
    def kron(self, left: Any, right: Any) -> Any: ...

    @abstractmethod
    def eye(self, dim: int, like: Any) -> Any: ...

    @abstractmethod
    def reshape(self, obj: Any, shape: Sequence[int], order: str = "C") -> Any: ...

    @abstractmethod
    def transpose(self, obj: Any, axes: Sequence[int] | None = None) -> Any: ...

    @abstractmethod
    def matmul(self, left: Any, right: Any) -> Any: ...

    def to_sparse(self, obj: Any) -> sp.sparray:
        raise NotImplementedError(f"{self.kind} backend does not support sparse conversion")

    @abstractmethod
    def shape(self, obj: Any) -> tuple[int, ...]: ...


class NumPyAdapter(BackendAdapter):
    kind = "dense"
    priority = 10

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, np.ndarray)

    def asarray(self, obj: Any) -> np.ndarray:
        return np.asarray(obj)

    def kron(self, left: Any, right: Any) -> np.ndarray:
        return np.kron(np.asarray(left), np.asarray(right))

    def eye(self, dim: int, like: Any) -> np.ndarray:
        return np.eye(dim, dtype=np.result_type(like))

    def reshape(self, obj: Any, shape: Sequence[int], order: str = "C") -> np.ndarray:
        return np.asarray(obj).reshape(tuple(shape), order=order)

    def transpose(self, obj: Any, axes: Sequence[int] | None = None) -> np.ndarray:
        return np.transpose(np.asarray(obj), axes=axes)

    def matmul(self, left: Any, right: Any) -> np.ndarray:
        return np.asarray(left) @ np.asarray(right)

    def shape(self, obj: Any) -> tuple[int, ...]:
        return tuple(int(dim) for dim in np.asarray(obj).shape)


class SciPySparseAdapter(BackendAdapter):
    kind = "sparse"
    priority = 20

    def can_handle(self, obj: Any) -> bool:
        return sp.issparse(obj)

    def asarray(self, obj: Any) -> sp.sparray:
        return to_sparse_array(obj)

    def kron(self, left: Any, right: Any) -> sp.sparray:
        left_sparse = to_sparse_array(left).asformat("coo") if sp.issparse(left) else left
        right_sparse = to_sparse_array(right).asformat("coo") if sp.issparse(right) else right
        return to_sparse_array(sp.kron(left_sparse, right_sparse))

    def eye(self, dim: int, like: Any) -> sp.sparray:
        return to_sparse_array(sp.eye_array(dim))

    def reshape(self, obj: Any, shape: Sequence[int], order: str = "C") -> sp.sparray:
        if order != "C":
            raise NotImplementedError("sparse reshape only supports C order")
        return to_sparse_array(to_sparse_array(obj).reshape(tuple(shape)))

    def transpose(self, obj: Any, axes: Sequence[int] | None = None) -> sp.sparray:
        if axes is not None:
            raise NotImplementedError("sparse backend does not support axis-wise transpose")
        return to_sparse_array(to_sparse_array(obj).transpose())

    def matmul(self, left: Any, right: Any) -> sp.sparray:
        return to_sparse_array(left @ right)

    def to_sparse(self, obj: Any) -> sp.sparray:
        return to_sparse_array(obj)

    def shape(self, obj: Any) -> tuple[int, ...]:
        return tuple(int(dim) for dim in to_sparse_array(obj).shape)


class CVXPYAdapter(BackendAdapter):
    kind = "symbolic"
    priority = 30

    def can_handle(self, obj: Any) -> bool:
        return isinstance(obj, cp.Expression)

    def asarray(self, obj: Any) -> cp.Expression:
        if isinstance(obj, cp.Expression):
            return obj
        if sp.issparse(obj):
            return cp.Constant(to_sparse_array(obj).toarray())
        return cp.Constant(np.asarray(obj))

    def kron(self, left: Any, right: Any) -> cp.Expression:
        return cp.kron(self.asarray(left), self.asarray(right))

    def eye(self, dim: int, like: Any) -> sp.sparray:
        return to_sparse_array(sp.eye_array(dim))

    def reshape(self, obj: Any, shape: Sequence[int], order: str = "C") -> cp.Expression:
        return cp.reshape(self.asarray(obj), tuple(shape), order=order)

    def transpose(self, obj: Any, axes: Sequence[int] | None = None) -> cp.Expression:
        if axes is not None:
            raise NotImplementedError("CVXPY backend does not support axis-wise transpose")
        return self.asarray(obj).T

    def matmul(self, left: Any, right: Any) -> cp.Expression:
        return self.asarray(left) @ self.asarray(right)

    def shape(self, obj: Any) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self.asarray(obj).shape)
