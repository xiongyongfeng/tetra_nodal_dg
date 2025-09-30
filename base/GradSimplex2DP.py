import numpy as np
from .JacobiP import jacobi_p
from .GradJacobiP import grad_jacobi_p


def grad_simplex_2dp(a, b, id, jd):
    """
    Return the derivatives of the modal basis (id,jd) on the 2D simplex at (a,b).

    Parameters:
    a, b : array_like
        Coordinates in the reference element
    id, jd : int
        Modal basis indices

    Returns:
    dmodedr, dmodeds : ndarray
        Derivatives of the modal basis with respect to r and s
    """
    # Calculate basis functions and their derivatives
    fa = jacobi_p(a, 0, 0, id)
    dfa = grad_jacobi_p(a, 0, 0, id)
    gb = jacobi_p(b, 2 * id + 1, 0, jd)
    dgb = grad_jacobi_p(b, 2 * id + 1, 0, jd)

    # r-derivative: d/dr = da/dr d/da + db/dr d/db = (2/(1-s)) d/da = (2/(1-b)) d/da
    dmodedr = dfa * gb

    if id > 0:
        dmodedr = dmodedr * ((0.5 * (1 - b)) ** (id - 1))

    # s-derivative: d/ds = ((1+a)/2)/((1-b)/2) d/da + d/db
    dmodeds = dfa * (gb * (0.5 * (1 + a)))

    if id > 0:
        dmodeds = dmodeds * ((0.5 * (1 - b)) ** (id - 1))

    tmp = dgb * ((0.5 * (1 - b)) ** id)

    if id > 0:
        tmp = tmp - 0.5 * id * gb * ((0.5 * (1 - b)) ** (id - 1))

    dmodeds = dmodeds + fa * tmp

    # Normalize
    dmodedr = (2 ** (id + 0.5)) * dmodedr
    dmodeds = (2 ** (id + 0.5)) * dmodeds

    return dmodedr, dmodeds
