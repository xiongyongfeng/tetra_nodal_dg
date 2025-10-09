from base.JacobiGL import jacobi_gl
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


def plot2d(r, s, N):

    # 2. 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(8, 8))  # 创建正方形画布，与图中比例类似
    fig.patch.set_facecolor("white")  # 设置图形背景为白色（通常默认就是白色）

    # 3. 绘制散点
    ax.scatter(r[:], s[:], marker="o", color="red", s=72, label="Group B (+)")
    plt.plot([x_gl[0], x_gl[N]], [-1, -1], color="black")
    plt.plot([-1, -1], [x_gl[0], x_gl[N]], color="black")
    plt.plot([x_gl[0], x_gl[N]], [x_gl[N], x_gl[0]], color="black")

    for i in range(1, N):
        plt.plot([-1, x_gl[N - i]], [x_gl[i], x_gl[i]], color="gray", linestyle="--")
    for i in range(1, N):
        plt.plot([x_gl[i], x_gl[i]], [-1, x_gl[N - i]], color="gray", linestyle="--")
    for i in range(1, N):
        plt.plot([-1, x_gl[i]], [x_gl[i], -1], color="gray", linestyle="--")
    # 4. 美化图表，使其更接近您描述的图
    # 设置坐标轴范围
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    # 设置坐标轴标签
    ax.set_xlabel("r")
    ax.set_ylabel("s")
    # 添加网格线，使观察分布更容易（参考图中似乎没有网格，可根据需要添加或删除）
    # ax.grid(True, linestyle='--', alpha=0.5)
    # 设置图表标题
    ax.set_title("(a) Scatter Plot of r and s")

    # 确保图形的宽高相等，使得散点分布不会因坐标轴缩放而变形
    ax.set_aspect("equal")

    # 显示图例（如果不需要可以注释掉）
    ax.legend()

    # 紧凑布局
    plt.tight_layout()

    # 显示图形
    plt.savefig("2d_lgl_points_set.png")


# 示例使用
if __name__ == "__main__":
    # 测试 JacobiGL 函数
    alpha, beta, N = 0, 0, 4  # 对应 Legendre 多项式
    x_gl = jacobi_gl(alpha, beta, N)

    print("x_gl:", x_gl)

    x_gl01 = (x_gl + 1) / 2
    print("x_gl01:", x_gl01)

    # 2D
    idx = 0
    NSP2D = (N + 1) * (N + 2) // 2
    r = np.zeros(NSP2D)
    s = np.zeros(NSP2D)
    for i in range(N + 1):
        for j in range(N + 1 - i):
            k = N - i - j
            print(f"i={i},j={j},k={k}")
            r[idx] = (1 + 2 * x_gl01[i] - x_gl01[j] - x_gl01[k]) / 3
            s[idx] = (1 - x_gl01[i] + 2 * x_gl01[j] - x_gl01[k]) / 3
            idx += 1
    r = r * 2 - 1
    s = s * 2 - 1
    print(f"r={r}")
    print(f"s={s}")
    plot2d(r, s, N)

    # 绘制这些点
    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(10, 2))
    # plt.plot(x_gl, np.zeros_like(x_gl), "o", markersize=8)
    # plt.title(f"Jacobi-Gauss-Lobatto 点 (N={N}, α={alpha}, β={beta})")
    # plt.xlim(-1.1, 1.1)
    # plt.yticks([])
    # plt.grid(True, alpha=0.3)
    # plt.show()
