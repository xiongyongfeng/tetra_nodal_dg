import numpy as np
from .JacobiP import jacobi_p


def vandermonde_1d(N, r):
    """
    Initialize the 1D Vandermonde Matrix, V_{ij} = phi_j(r_i)

    Parameters:
    N : int
        Polynomial order
    r : array_like
        Coordinates at which to evaluate the polynomials

    Returns:
    V1D : ndarray
        1D Vandermonde matrix of shape (len(r), N+1)
    """
    # Convert input to numpy array for consistent handling
    r = np.asarray(r)

    # Initialize the Vandermonde matrix with zeros
    V1D = np.zeros((len(r), N + 1))

    # Fill the Vandermonde matrix
    for j in range(N + 1):
        V1D[:, j] = jacobi_p(r, 0, 0, j)

    return V1D
