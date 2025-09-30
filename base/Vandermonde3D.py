import numpy as np
from .Simplex3DP import simplex_3d_p
from .RSTtoABC import rsttoabc


def vandermonde_3d(N, r, s, t):
    """
    Initialize the 3D Vandermonde Matrix
    V_{ij} = phi_j(r_i, s_i, t_i)

    Parameters:
    N : int
        Polynomial order
    r, s, t : array_like
        Coordinate points

    Returns:
    V3D : ndarray
        3D Vandermonde matrix of shape (len(r), (N+1)*(N+2)*(N+3)//6)
    """

    r = np.asarray(r)
    s = np.asarray(s)
    t = np.asarray(t)

    # Calculate number of basis functions
    n_basis = (N + 1) * (N + 2) * (N + 3) // 6
    V3D = np.zeros((len(r), n_basis))

    # Transfer to (a,b,c) coordinates - need to implement rsttoabc function
    a, b, c = rsttoabc(r, s, t)

    # Build the Vandermonde matrix
    sk = 0
    for i in range(0, N + 1):
        for j in range(0, N - i + 1):
            for k in range(0, N - i - j + 1):
                V3D[:, sk] = simplex_3d_p(a, b, c, i, j, k)
                sk += 1

    return V3D
