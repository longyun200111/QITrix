from typing import Any, Literal, overload

import scipy.sparse as sp

from QITrix._internal.types import NDArray, SparseArray


@overload
def fock(dim: int, n: int) -> NDArray[Any]: ...


@overload
def fock(dim: int, n: int, format: Literal["dense"]) -> NDArray[Any]: ...


@overload
def fock(dim: int, n: int, format: Literal["bsr"]) -> sp.bsr_array: ...


@overload
def fock(dim: int, n: int, format: Literal["coo"]) -> sp.coo_array: ...


@overload
def fock(dim: int, n: int, format: Literal["csc"]) -> sp.csc_array: ...


@overload
def fock(dim: int, n: int, format: Literal["csr"]) -> sp.csr_array: ...


@overload
def fock(dim: int, n: int, format: Literal["dia"]) -> sp.dia_array: ...


@overload
def fock(dim: int, n: int, format: Literal["dok"]) -> sp.dok_array: ...


@overload
def fock(dim: int, n: int, format: Literal["lil"]) -> sp.lil_array: ...


def fock(dim: int, n: int, format: str = "dense") -> NDArray[Any] | SparseArray:
    if dim < 1:
        raise ValueError("dim must be positive")
    if n < 0 or n >= dim:
        raise ValueError("n must satisfy 0 <= n < dim")
    state = sp.coo_array(([1.0], ([n], [0])), shape=(dim, 1))
    if format == "dense":
        return state.toarray()
    return state.asformat(format)


@overload
def fock_dm(dim: int, n: int) -> NDArray[Any]: ...


@overload
def fock_dm(dim: int, n: int, format: Literal["dense"]) -> NDArray[Any]: ...


@overload
def fock_dm(dim: int, n: int, format: Literal["bsr"]) -> sp.bsr_array: ...


@overload
def fock_dm(dim: int, n: int, format: Literal["coo"]) -> sp.coo_array: ...


@overload
def fock_dm(dim: int, n: int, format: Literal["csc"]) -> sp.csc_array: ...


@overload
def fock_dm(dim: int, n: int, format: Literal["csr"]) -> sp.csr_array: ...


@overload
def fock_dm(dim: int, n: int, format: Literal["dia"]) -> sp.dia_array: ...


@overload
def fock_dm(dim: int, n: int, format: Literal["dok"]) -> sp.dok_array: ...


@overload
def fock_dm(dim: int, n: int, format: Literal["lil"]) -> sp.lil_array: ...


def fock_dm(dim: int, n: int, format: str = "dense") -> NDArray[Any] | SparseArray:
    if dim < 1:
        raise ValueError("dim must be positive")
    if n < 0 or n >= dim:
        raise ValueError("n must satisfy 0 <= n < dim")
    state = sp.coo_array(([1.0], ([n], [n])), shape=(dim, dim))
    if format == "dense":
        return state.toarray()
    return state.asformat(format)
