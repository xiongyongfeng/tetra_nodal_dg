import numpy as np
from .GradSimplex3DP import grad_simplex_3dp
from .RSTtoABC import rsttoabc


def grad_vandermonde_3d(N, r, s, t):
    """
    Initialize the gradient of the modal basis (i,j,k) at (r,s,t) at order N

    Parameters:
    N: polynomial order
    r, s, t: coordinate arrays

    Returns:
    V3Dr, V3Ds, V3Dt: gradient matrices of the modal basis
    """

    # Get the number of points
    npoints = len(r)

    # Calculate the number of basis functions
    nbasis = (N + 1) * (N + 2) * (N + 3) // 6

    # Initialize output matrices
    V3Dr = np.zeros((npoints, nbasis))
    V3Ds = np.zeros((npoints, nbasis))
    V3Dt = np.zeros((npoints, nbasis))

    # Convert to tensor-product coordinates
    a, b, c = rsttoabc(r, s, t)

    # Initialize sk counter
    sk = 0

    # Loop through polynomial orders
    for i in range(0, N + 1):
        for j in range(0, N - i + 1):
            for k in range(0, N - i - j + 1):
                # Get gradients for this basis function
                dr, ds, dt = grad_simplex_3dp(a, b, c, i, j, k)

                # Store in output matrices
                V3Dr[:, sk] = dr
                V3Ds[:, sk] = ds
                V3Dt[:, sk] = dt

                sk += 1

    return V3Dr, V3Ds, V3Dt
