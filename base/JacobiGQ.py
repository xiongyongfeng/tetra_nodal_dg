import numpy as np
from scipy.special import gamma
from scipy.linalg import eig


def jacobi_gq(alpha, beta, N):
    """
    计算 N 阶 Gauss 求积点和权重，基于 Jacobi 多项式 (alpha, beta > -1)

    参数:
        alpha: Jacobi 多项式参数,alpha > -1
        beta: Jacobi 多项式参数,beta > -1
        N: 求积点个数

    返回:
        x: 求积点，长度为 N+1 的数组
        w: 权重，长度为 N+1 的数组
    """
    # 处理 N=0 的特殊情况
    if N == 0:
        x = np.array([(alpha - beta) / (alpha + beta + 2)])
        w = np.array([2.0])
        return x, w

    # 构建对称矩阵 J
    J = np.zeros((N + 1, N + 1))

    # 计算对角线元素
    h1 = 2 * np.arange(0, N + 1) + alpha + beta
    diag_vals = -0.5 * (alpha**2 - beta**2) / ((h1 + 2) * h1)
    np.fill_diagonal(J, diag_vals)

    # 计算次对角线元素
    indices = np.arange(1, N + 1)
    subdiag_vals = (
        2
        / (h1[0:N] + 2)
        * np.sqrt(
            indices
            * (indices + alpha + beta)
            * (indices + alpha)
            * (indices + beta)
            / ((h1[0:N] + 1) * (h1[0:N] + 3))
        )
    )

    # 设置次对角线
    for i in range(N):
        J[i, i + 1] = subdiag_vals[i]
        J[i + 1, i] = subdiag_vals[i]  # 使矩阵对称

    # 特殊处理：当 alpha+beta 接近零时
    if alpha + beta < 10 * np.finfo(float).eps:
        J[0, 0] = 0.0

    # 通过特征值求解计算求积点
    D, V = eig(J)  # D 是特征值，V 是特征向量
    x = np.real(D)  # 取实部

    # 计算权重
    w = (V[0, :] ** 2) * (
        2 ** (alpha + beta + 1)
        / (alpha + beta + 1)
        * gamma(alpha + 1)
        * gamma(beta + 1)
        / gamma(alpha + beta + 1)
    )
    w = np.real(w)  # 取实部

    # 对求积点和权重进行排序
    idx = np.argsort(x)
    x = x[idx]
    w = w[idx]

    return x, w


# 示例使用
if __name__ == "__main__":
    # 测试代码
    alpha, beta, N = 0, 0, 1  # 对应 Legendre 多项式
    x, w = jacobi_gq(alpha, beta, N)

    print("求积点 (x):", x)
    print("权重 (w):", w)

    # 验证权重和是否为 2 (对于 [-1,1] 区间)
    print("权重和:", np.sum(w))
