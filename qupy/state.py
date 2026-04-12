import scipy.sparse as sp

def fock(dim: int, n: int, format: str = "array"):
    """
    Return the Fock basis ket ``|n>`` in a finite-dimensional Hilbert space.

    Parameters
    ----------
    dim : int
        Hilbert-space dimension.
    n : int
        Occupation-number index.
    format : str, optional
        Output sparse format understood by SciPy.

    Returns
    -------
    scipy.sparse.spmatrix | sp.sparray
        Sparse column vector representing ``|n>``.
    """
    if n >= dim:
        raise ValueError("n must be less than dim")
    return sp.coo_array(([1.], ([n], [0])), shape=(dim, 1)).asformat(format)

def fock_dm(dim: int, n: int, format: str = "array"):
    """
    Return the projector ``|n><n|`` in a finite-dimensional Hilbert space.

    Parameters
    ----------
    dim : int
        Hilbert-space dimension.
    n : int
        Occupation-number index.
    format : str, optional
        Output sparse format understood by SciPy.

    Returns
    -------
    scipy.sparse.spmatrix | sp.sparray
        Sparse density matrix of the Fock state ``|n>``.
    """
    if n >= dim:
        raise ValueError("n must be less than dim")
    return sp.coo_array(([1.], ([n], [n])), shape=(dim, dim)).asformat(format)
