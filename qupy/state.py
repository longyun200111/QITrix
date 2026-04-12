from typing import Any, Literal, overload

import scipy.sparse as sp

from ._types import NDArray, SparseArray

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
    """
    Return the Fock basis ket ``|n>`` in a finite-dimensional Hilbert space.

    Parameters
    ----------
    dim : int
        Hilbert-space dimension.
    n : int
        Occupation-number index.
    format : str, optional
        Output format. Use ``"dense"`` for a NumPy array, or any sparse format
        name understood by SciPy ``asformat`` such as ``"csr"`` or ``"coo"``.

    Returns
    -------
    np.ndarray | sp.sparray
        Column vector representing ``|n>``. The return type is dense if
        ``format == "dense"``, and sparse otherwise.
    """
    if n >= dim:
        raise ValueError("n must be less than dim")
    state = sp.coo_array(([1.], ([n], [0])), shape=(dim, 1))
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
    """
    Return the projector ``|n><n|`` in a finite-dimensional Hilbert space.

    Parameters
    ----------
    dim : int
        Hilbert-space dimension.
    n : int
        Occupation-number index.
    format : str, optional
        Output format. Use ``"dense"`` for a NumPy array, or any sparse format
        name understood by SciPy ``asformat`` such as ``"csr"`` or ``"coo"``.

    Returns
    -------
    np.ndarray | sp.sparray
        Projector ``|n><n|``. The return type is dense if
        ``format == "dense"``, and sparse otherwise.
    """
    if n >= dim:
        raise ValueError("n must be less than dim")
    state = sp.coo_array(([1.], ([n], [n])), shape=(dim, dim))
    if format == "dense":
        return state.toarray()
    return state.asformat(format)
