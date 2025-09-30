import numpy as np
from .JacobiP import jacobi_p
from .GradJacobiP import grad_jacobi_p


def grad_simplex_3dp(a, b, c, id, jd, kd):
    """
    Return the derivatives of the modal basis (id,jd,kd)
    on the 3D simplex at (a,b,c)

    Parameters:
    a, b, c: coordinates in the simplex
    id, jd, kd: modal basis indices

    Returns:
    V3Dr, V3Ds, V3Dt: derivatives in r, s, t directions
    """

    # 假设已经实现了相应的Jacobi多项式函数
    # jacobi_p(x, alpha, beta, n) 和 grad_jacobi_p(x, alpha, beta, n)
    fa = jacobi_p(a, 0, 0, id)
    dfa = grad_jacobi_p(a, 0, 0, id)
    gb = jacobi_p(b, 2 * id + 1, 0, jd)
    dgb = grad_jacobi_p(b, 2 * id + 1, 0, jd)
    hc = jacobi_p(c, 2 * (id + jd) + 2, 0, kd)
    dhc = grad_jacobi_p(c, 2 * (id + jd) + 2, 0, kd)

    # r-derivative
    V3Dr = dfa * (gb * hc)
    if id > 0:
        V3Dr = V3Dr * ((0.5 * (1 - b)) ** (id - 1))
    if id + jd > 0:
        V3Dr = V3Dr * ((0.5 * (1 - c)) ** (id + jd - 1))

    # s-derivative
    V3Ds = 0.5 * (1 + a) * V3Dr

    tmp = dgb * ((0.5 * (1 - b)) ** id)
    if id > 0:
        tmp = tmp + (-0.5 * id) * (gb * (0.5 * (1 - b)) ** (id - 1))
    if id + jd > 0:
        tmp = tmp * ((0.5 * (1 - c)) ** (id + jd - 1))

    tmp = fa * (tmp * hc)
    V3Ds = V3Ds + tmp

    # t-derivative
    V3Dt = 0.5 * (1 + a) * V3Dr + 0.5 * (1 + b) * tmp

    tmp = dhc * ((0.5 * (1 - c)) ** (id + jd))
    if id + jd > 0:
        tmp = tmp - 0.5 * (id + jd) * (hc * ((0.5 * (1 - c)) ** (id + jd - 1)))

    tmp = fa * (gb * tmp)
    tmp = tmp * ((0.5 * (1 - b)) ** id)
    V3Dt = V3Dt + tmp

    # normalize
    factor = 2 ** (2 * id + jd + 1.5)
    V3Dr = V3Dr * factor
    V3Ds = V3Ds * factor
    V3Dt = V3Dt * factor

    return V3Dr, V3Ds, V3Dt
