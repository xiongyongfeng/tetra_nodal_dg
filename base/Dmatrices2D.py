import numpy as np
from .GradVandermonde2D import grad_vandermonde_2d


def dmatrices_2d(N, r, s, V):
    """
    Initialize the (r,s) differentiation matrices on the simplex,
    evaluated at (r,s) at order N

    Parameters:
    N : int
        Polynomial order
    r : array_like
        First set of coordinates
    s : array_like
        Second set of coordinates
    V : ndarray
        Vandermonde matrix

    Returns:
    Dr, Ds : tuple of ndarrays
        Differentiation matrices with respect to r and s
    """
    Vr, Vs = grad_vandermonde_2d(N, r, s)

    # Solve V * Dr^T = Vr^T and V * Ds^T = Vs^T
    # Using least squares solution for better numerical stability
    Dr = np.linalg.lstsq(V.T, Vr.T, rcond=None)[0].T
    Ds = np.linalg.lstsq(V.T, Vs.T, rcond=None)[0].T

    return Dr, Ds
