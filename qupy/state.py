import scipy.sparse as sp
import numpy as np

def fock(dim: int, n: int, format: str = "dense") -> np.ndarray | sp.spmatrix | sp.sparray:
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
    np.ndarray | scipy.sparse.spmatrix | sp.sparray
        Column vector representing ``|n>``. The return type is dense if
        ``format == "dense"``, and sparse otherwise.
    """
    if n >= dim:
        raise ValueError("n must be less than dim")
    state = sp.coo_array(([1.], ([n], [0])), shape=(dim, 1))
    if format == "dense":
        return state.toarray()
    return state.asformat(format)

def fock_dm(dim: int, n: int, format: str = "dense") -> np.ndarray | sp.spmatrix | sp.sparray:
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
    np.ndarray | scipy.sparse.spmatrix | sp.sparray
        Projector ``|n><n|``. The return type is dense if
        ``format == "dense"``, and sparse otherwise.
    """
    if n >= dim:
        raise ValueError("n must be less than dim")
    state = sp.coo_array(([1.], ([n], [n])), shape=(dim, dim))
    if format == "dense":
        return state.toarray()
    return state.asformat(format)
