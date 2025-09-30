import numpy as np
from .JacobiGQ import jacobi_gq


def jacobi_gl(alpha, beta, N):
    """
    计算 N 阶 Gauss-Lobatto 求积点，基于 Jacobi 多项式 (alpha, beta > -1)

    参数:
        alpha: Jacobi 多项式参数,alpha > -1
        beta: Jacobi 多项式参数,beta > -1
        N: 求积点个数减一(返回的数组长度为 N+1)

    返回:
        x: Gauss-Lobatto 求积点，长度为 N+1 的数组，包含端点 -1 和 1
    """
    # 处理 N=1 的特殊情况
    if N == 1:
        return np.array([-1.0, 1.0])

    # 调用 JacobiGQ 函数获取内部点（需要先实现 jacobi_gq 函数）
    xint, w = jacobi_gq(alpha + 1, beta + 1, N - 2)

    # 构造完整的 Gauss-Lobatto 点集（包含端点 -1 和 1）
    x = np.concatenate(([-1.0], xint, [1.0]))

    return x


# 注意：此函数依赖于之前实现的 jacobi_gq 函数
# 以下是完整的实现示例，包含 jacobi_gq 函数


# 示例使用
if __name__ == "__main__":
    # 测试 JacobiGL 函数
    alpha, beta, N = 0, 0, 4  # 对应 Legendre 多项式
    x_gl = jacobi_gl(alpha, beta, N)

    print("Gauss-Lobatto 求积点 (x):", x_gl)

    # 绘制这些点
    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(10, 2))
    # plt.plot(x_gl, np.zeros_like(x_gl), "o", markersize=8)
    # plt.title(f"Jacobi-Gauss-Lobatto 点 (N={N}, α={alpha}, β={beta})")
    # plt.xlim(-1.1, 1.1)
    # plt.yticks([])
    # plt.grid(True, alpha=0.3)
    # plt.show()
