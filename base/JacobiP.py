import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt  # 导入绘图库


def jacobi_p(x, alpha, beta, N):
    """
    计算归一化正交的雅可比多项式 P_N^{(alpha, beta)}(x) 的值

    参数:
        x: 计算点，可以是标量或数组
        alpha: 参数，alpha > -1
        beta: 参数，beta > -1
        N: 多项式的阶数

    返回:
        P: 在点 x 处的函数值，形状与 x 相同
    """
    # 将 x 转换为 numpy 数组并确保其为行向量风格（1维数组）
    xp = np.array(x)
    orig_shape = xp.shape
    xp = xp.flatten()  # 展平以便于处理，最后再重塑回原始形状

    # 初始化一个矩阵来存储所有阶数（从0到N）的多项式值
    PL = np.zeros((N + 1, len(xp)))

    # 计算初始值 P0
    gamma0 = (
        2 ** (alpha + beta + 1)
        / (alpha + beta + 1)
        * gamma(alpha + 1)
        * gamma(beta + 1)
        / gamma(alpha + beta + 1)
    )
    PL[0, :] = 1.0 / np.sqrt(gamma0)

    if N == 0:
        return PL[0, :].reshape(orig_shape)  # 返回第0阶结果

    # 计算 P1
    gamma1 = (alpha + 1) * (beta + 1) / (alpha + beta + 3) * gamma0
    PL[1, :] = ((alpha + beta + 2) * xp / 2 + (alpha - beta) / 2) / np.sqrt(gamma1)

    if N == 1:
        return PL[1, :].reshape(orig_shape)  # 返回第1阶结果

    # 递归计算的系数初始值 (for i=1)
    aold = (
        2 / (2 + alpha + beta) * np.sqrt((alpha + 1) * (beta + 1) / (alpha + beta + 3))
    )

    # 使用递推关系从 i=1 到 i=N-1 进行计算
    for i in range(1, N):  # i 从 1 到 N-1 (对应MATLAB中的 i=1:N-1)
        h1 = 2 * i + alpha + beta
        # 计算新的递推系数 anew 和 bnew
        anew = (
            2
            / (h1 + 2)
            * np.sqrt(
                (i + 1)
                * (i + 1 + alpha + beta)
                * (i + 1 + alpha)
                * (i + 1 + beta)
                / (h1 + 1)
                / (h1 + 3)
            )
        )
        bnew = -(alpha**2 - beta**2) / h1 / (h1 + 2)

        # 三项递推关系核心公式:
        # P_{i+1} = (1/anew) * [ -aold * P_{i-1} + (x - bnew) * P_{i} ]
        PL[i + 1, :] = (1.0 / anew) * (-aold * PL[i - 1, :] + (xp - bnew) * PL[i, :])

        aold = anew  # 更新系数用于下一次迭代

    # 返回第N阶的结果，并重塑为输入x的形状
    P_N = PL[N, :].reshape(orig_shape)
    return P_N


from sympy import symbols, diff, factorial, lambdify


def jacobi_p_rodrigues(x, n, alpha, beta):
    """
    https://en.wikipedia.org/wiki/Jacobi_polynomials
    使用 Rodrigues 公式计算雅可比多项式的值（符号计算）
    参数:
        x: 自变量（符号）
        n: 多项式阶数
        alpha, beta: 参数
    返回:
        P_n^{(alpha,beta)}(x) 的表达式
    """
    if n == 0:
        # 如果x是数组/向量，返回相同维度的1；如果是符号变量，返回标量1
        if hasattr(x, "__len__"):
            return np.ones_like(x)
        return 1
    # Rodrigues 公式
    factor = (
        (-1) ** n / (2**n * factorial(n)) * (1 - x) ** (-alpha) * (1 + x) ** (-beta)
    )
    inside = (1 - x) ** (alpha + n) * (1 + x) ** (beta + n)
    derivative = diff(inside, x, n)  # 求n阶导数

    return factor * derivative


# 示例用法和绘图
if __name__ == "__main__":
    x_points = np.linspace(-1, 1, 20)
    alpha_val = 2
    beta_val = 0
    N_val = 1

    result_jacobi_p = jacobi_p(x_points, alpha_val, beta_val, N_val)
    print(f"在点 {x_points} 上的 {N_val} 阶雅可比多项式值:")
    print(f"result={result_jacobi_p}")

    # 创建符号变量
    x_sym = symbols("x")

    # 计算雅可比多项式表达式
    expr = jacobi_p_rodrigues(x_sym, N_val, alpha_val, beta_val)
    if N_val > 0:
        print(f"P_{N_val}^{({alpha_val},{beta_val})}(x) = {expr.simplify()}")

    # 将符号表达式转换为数值函数
    x_vals = np.linspace(-1, 1, 100)  # 在区间[-1, 1]上采样
    print(x_vals)
    f_func = lambdify(x_sym, expr, "numpy")
    y_vals = f_func(x_vals)
    if np.isscalar(y_vals):
        y_vals = np.full_like(x_vals, y_vals)

    # 创建图形
    plt.figure(figsize=(10, 6))  # 设置图形大小
    plt.plot(
        x_points,
        result_jacobi_p,
        "b-",
        linewidth=2,
        label=f"Jacobi Polynomial Orthonormalized (N={N_val})",
    )  # 绘制线图
    plt.scatter(
        x_points, result_jacobi_p, color="blue", s=50, zorder=5
    )  # 在数据点上添加红色圆点
    plt.scatter(
        x_vals,
        y_vals,
        color="black",
        s=50,
        zorder=5,
        label=f"Jacobi Polynomial Rodrigues (N={N_val})",
    )
    # 添加标题和标签
    plt.title(
        f"Jacobi Polynomial $P_{{{N_val}}}^{{( {alpha_val}, {beta_val} )}}$(x) on [-1, 1]",
        fontsize=14,
    )
    plt.xlabel("x")
    plt.ylabel(f"$P_{{{N_val}}}^{{( {alpha_val}, {beta_val} )}}$(x)")

    # 添加网格和图例
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 保存图像为PNG格式[1,2](@ref)
    plt.savefig(
        "jacobi_polynomial_plot.png", dpi=300, bbox_inches="tight"
    )  # 高分辨率保存，去除多余空白[1,2](@ref)
