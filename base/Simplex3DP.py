import numpy as np

from .JacobiP import jacobi_p


def simplex_3d_p(a, b, c, i, j, k):
    """
    在单纯形上计算二维正交多项式在点(a,b,c)处阶数为(i,j,k)的值

    参数:
        a: 第一个坐标值（可以是标量或数组）
        b: 第二个坐标值（可以是标量或数组）
        c: 第三个坐标值（可以是标量或数组）
        i: 第一个方向的阶数
        j: 第二个方向的阶数
        k: 第三个方向的阶数

    返回:
        P: 在点(a,b)处的二维正交多项式值
    """
    # 计算第一个方向的雅可比多项式
    h1 = jacobi_p(a, 0, 0, i)

    # 计算第二个方向的雅可比多项式
    h2 = jacobi_p(b, 2 * i + 1, 0, j)

    h3 = jacobi_p(c, 2 * i + 2 * j + 2, 0, k)

    # 计算最终结果
    P = 2 * np.sqrt(2.0) * h1 * h2 * (1 - b) ** i * h3 * (1 - c) ** (i + j)

    return P
