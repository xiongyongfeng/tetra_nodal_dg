import numpy as np
from .RStoAB import rstoab
from .GradSimplex2DP import grad_simplex_2dp


def grad_vandermonde_2d(N, r, s):
    """
    Initialize the gradient of the modal basis (i,j) at (r,s) at order N

    Parameters:
    N : int
        Polynomial order
    r : array_like
        First set of coordinates
    s : array_like
        Second set of coordinates

    Returns:
    V2Dr, V2Ds : tuple of ndarrays
        Gradients of the 2D Vandermonde matrix with respect to r and s
    """
    n_points = len(r)
    n_modes = (N + 1) * (N + 2) // 2

    # Initialize matrices
    V2Dr = np.zeros((n_points, n_modes))
    V2Ds = np.zeros((n_points, n_modes))

    # find tensor-product coordinates
    a, b = rstoab(r, s)

    # Initialize matrices
    sk = 0  # Python uses 0-based indexing
    for i in range(0, N + 1):
        for j in range(0, N - i + 1):
            V2Dr[:, sk], V2Ds[:, sk] = grad_simplex_2dp(a, b, i, j)
            sk += 1

    return V2Dr, V2Ds
