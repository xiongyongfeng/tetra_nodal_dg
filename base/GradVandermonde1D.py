import numpy as np
from .GradJacobiP import grad_jacobi_p


def grad_vandermonde_1d(N, r):
    """
    初始化一维模态基的梯度

    Parameters:
    N : int
        多项式的最高阶数
    r : array_like
        计算点的坐标

    Returns:
    DVr : ndarray
        梯度矩阵，形状为 (len(r), N+1)
    """
    r = np.asarray(r)
    DVr = np.zeros((len(r), N + 1))

    # 初始化矩阵
    for i in range(N + 1):
        DVr[:, i] = grad_jacobi_p(r, 0, 0, i)

    return DVr
