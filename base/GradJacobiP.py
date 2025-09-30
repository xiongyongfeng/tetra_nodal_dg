import numpy as np
from .JacobiP import jacobi_p


def grad_jacobi_p(r, alpha, beta, N):
    """
    计算雅可比多项式的导数

    Parameters:
    r : array_like
        计算点的坐标
    alpha : float
        雅可比多项式的alpha参数，需满足alpha > -1
    beta : float
        雅可比多项式的beta参数，需满足beta > -1
    N : int
        多项式的阶数

    Returns:
    dP : ndarray
        雅可比多项式在点r处的导数值
    """
    r = np.asarray(r)
    dP = np.zeros_like(r)

    if N == 0:
        dP[:] = 0.0
    else:
        dP = np.sqrt(N * (N + alpha + beta + 1)) * jacobi_p(
            r, alpha + 1, beta + 1, N - 1
        )

    return dP
