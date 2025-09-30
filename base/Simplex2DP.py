import numpy as np

from .JacobiP import jacobi_p


def simplex_2d_p(a, b, i, j):
    """
    在单纯形上计算二维正交多项式在点(a,b)处阶数为(i,j)的值

    参数:
        a: 第一个坐标值（可以是标量或数组）
        b: 第二个坐标值（可以是标量或数组）
        i: 第一个方向的阶数
        j: 第二个方向的阶数

    返回:
        P: 在点(a,b)处的二维正交多项式值
    """
    # 计算第一个方向的雅可比多项式
    h1 = jacobi_p(a, 0, 0, i)

    # 计算第二个方向的雅可比多项式
    h2 = jacobi_p(b, 2 * i + 1, 0, j)

    # 计算最终结果
    P = np.sqrt(2.0) * h1 * h2 * (1 - b) ** i

    return P


# 示例使用
if __name__ == "__main__":
    # 测试 Simplex2DP 函数
    # 创建网格点

    N = 1
    for i in range(N + 1):
        for j in range(N + 1):
            if i + j <= N:
                m = j + (N + 1) * i + 1 - i * (i - 1) / 2
                print(f"i = {i}, j={j}, m = {m}")
    n_points = 5
    a_vals = np.linspace(-1, 1, n_points)
    b_vals = np.linspace(-1, 1, n_points * 2)
    A, B = np.meshgrid(a_vals, b_vals)
    print(f"A={A}")
    print(f"B={B}")

    # 计算单纯形上的二维正交多项式值
    i, j = 1, 0  # 阶数
    P_vals = simplex_2d_p(A, B, i, j)

    # 可视化结果
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 绘制曲面
    surf = ax.plot_surface(A, B, P_vals, cmap=cm.viridis, alpha=0.8)

    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # 设置标签和标题
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_zlabel(f"P_{i},{j}(a,b)")
    ax.set_title(f"2D Orthonormal Polynomial on Simplex (Order ({i},{j}))")

    plt.show()
