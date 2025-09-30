import numpy as np
from .Simplex2DP import simplex_2d_p
from .RStoAB import rstoab


def vandermonde_2d(N, r, s):
    """
    Initialize the 2D Vandermonde Matrix, V_{ij} = phi_j(r_i, s_i)

    Parameters:
    N : int
        Polynomial order
    r : array_like
        First set of coordinates
    s : array_like
        Second set of coordinates

    Returns:
    V2D : ndarray
        2D Vandermonde matrix
    """
    # Convert to (a,b) coordinates
    a, b = rstoab(r, s)

    # Initialize the Vandermonde matrix
    n_points = len(r)
    n_modes = (N + 1) * (N + 2) // 2
    V2D = np.zeros((n_points, n_modes))

    # Build the Vandermonde matrix
    sk = 0
    for i in range(0, N + 1):
        for j in range(0, N - i + 1):
            V2D[:, sk] = simplex_2d_p(a, b, i, j)
            sk += 1

    return V2D
