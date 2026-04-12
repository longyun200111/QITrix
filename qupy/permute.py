import cvxpy as cp
import numpy as np
import scipy.sparse as sp

def _permute_numpy(
        rho: np.ndarray, 
        dims: list[int], 
        perm: list[int], 
        direction: str = "both"
    ) -> np.ndarray:
    """
    Permute the axes of a density matrix using NumPy.

    Parameters
    ----------
    rho : np.ndarray
        Input operator written on the tensor-product space whose subsystem
        dimensions are given by ``dims``.
    dims : list[int]
        Dimensions of the subsystems in the same order as the tensor factors.
    perm : list[int]
        Permutation of subsystem indices. The indices are assumed to be already
        normalized to the range ``0, ..., len(dims) - 1``.
    direction : str, optional
        Direction of permutation. Must be one of "both", "left", or "right".
        Default is "both".

    Returns
    -------
    np.ndarray
        The permuted operator.

    Notes
    -----
    The function constructs the permutation matrix corresponding to the given
    permutation and applies it to the input operator from the left, right, or both sides.
    """
    dim = int(np.prod(dims, dtype=int)) if dims else 1
    if direction == "both":
        tensor_rho = np.asarray(rho).reshape(dims + dims)
        tensor_perm = perm + [p + len(dims) for p in perm]
        permuted_tensor_rho = np.transpose(tensor_rho, axes=tensor_perm)
        perm_rho = permuted_tensor_rho.reshape((dim, dim))
    elif direction == "left":
        tensor_rho = np.asarray(rho).reshape((*dims, -1))
        tensor_perm = perm + [len(dims)]
        permuted_tensor_rho = np.transpose(tensor_rho, axes=tensor_perm)
        perm_rho = permuted_tensor_rho.reshape((dim, -1))
    elif direction == "right":
        tensor_rho = np.asarray(rho).reshape((-1, *dims))
        tensor_perm = [0] + [p + 1 for p in perm]
        permuted_tensor_rho = np.transpose(tensor_rho, axes=tensor_perm)
        perm_rho = permuted_tensor_rho.reshape((-1, dim))
    else:
        raise ValueError("Invalid direction. Must be 'both', 'left', or 'right'.")

    return perm_rho

def _get_permutation_matrix(dims, perm):
    """
    Construct the subsystem permutation matrix associated with ``perm``.

    Parameters
    ----------
    dims : list[int]
        Dimensions of the subsystems.
    perm : list[int]
        Target ordering of subsystem indices.

    Returns
    -------
    sp.csr_array
        Sparse permutation matrix acting on the vectorized Hilbert space.
    """
    d = np.prod(dims)
    coords = np.arange(d)
    ravel_coords = np.reshape(coords, dims)
    perm_coords = np.transpose(ravel_coords, perm).flatten()
    perm_matrix = sp.coo_array(([1.]*d, (coords, perm_coords)), shape=(d, d)).tocsr()
    return perm_matrix

def permute(rho, dims, perm, direction: str = "both"):
    """
    Permute subsystem order in an operator or rectangular matrix.

    Parameters
    ----------
    rho : np.ndarray | sp.sparray | cp.Expression
        Matrix-like input object.
    dims : list[int]
        Dimensions of the subsystems being permuted.
    perm : list[int]
        Target subsystem ordering.
    direction : str, optional
        Which side of the matrix should be permuted. Supported values are
        ``"both"``, ``"left"``, and ``"right"``.

    Returns
    -------
    np.ndarray | sp.sparray | cp.Expression
        Permuted matrix in the same backend family as ``rho``.
    """
    if isinstance(rho, np.ndarray):
        return _permute_numpy(rho, dims, perm, direction)

    perm_matrix = _get_permutation_matrix(dims, perm)

    if direction == "both":
        perm_rho = perm_matrix @ rho @ perm_matrix.T
    elif direction == "left":
        perm_rho = perm_matrix @ rho
    elif direction == "right":
        perm_rho = rho @ perm_matrix.T
    else:
        raise ValueError("Invalid direction. Must be 'both', 'left', or 'right'.")
    
    return perm_rho
