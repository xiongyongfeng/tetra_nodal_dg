import numpy as np
from JacobiP import jacobi_p  # 请确保 JacbiP 模块已正确导入
from Simplex2DP import simplex_2d_p
import matplotlib.pyplot as plt
from matplotlib import cm
from RStoAB import rstoab


# 生成 (r, s) 坐标下的网格点，并过滤定义域
n_points = 10
r_vals = np.linspace(-1, 1, n_points)  # r 从 -1 到 1
s_vals = np.linspace(-1, 0.999, n_points)  # s 从 -1 到 0.999 (避免 s=1)
R, S = np.meshgrid(r_vals, s_vals)
print(f"R={R}")
print(f"S={S}")

# 应用约束条件 r + s <= 0, r>=-1, s>=-1 来创建掩膜
mask = (R + S <= 0) & (R >= -1) & (S >= -1)
print(f"mask={mask}")

# 初始化 P_vals_rs，用于存储 (r, s) 空间的值，初始化为 NaN
P_vals_rs = np.full_like(R, np.nan)

# 将掩膜应用于网格点，只处理定义域内的点
R_masked = R[mask]
S_masked = S[mask]
print(f"R_masked={R_masked}")
print(f"S_masked={S_masked}")

# 将 (r, s) 转换到 (a, b)
A_masked, B_masked = rstoab(R_masked, S_masked)
print(f"A_masked={A_masked}")
print(f"B_masked={B_masked}")

# 计算多项式值 (例如 i=0, j=3)
i, j = 1, 0
P_masked = simplex_2d_p(A_masked, B_masked, i, j)

# 将计算得到的值填充回 P_vals_rs 的对应位置
P_vals_rs[mask] = P_masked

# 可视化 (r, s) 坐标系下的曲面
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")

# 绘制曲面，使用 masked array 避免绘制无效点
surf = ax.plot_surface(
    R, S, P_vals_rs, cmap=cm.viridis, alpha=0.8, linewidth=0, antialiased=True
)

# 添加颜色条
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label=f"P_{i},{j}(r,s)")

# 设置标签和标题
ax.set_xlabel("r")
ax.set_ylabel("s")
ax.set_zlabel(f"P_{i},{j}(a,b)")
ax.set_title(f"2D Orthonormal Polynomial in (r, s) Coordinates (Order ({i},{j}))")

# 设置视角以便更好地观察
ax.view_init(elev=30, azim=45)

points_lists = [[-1, -1], [1, -1], [-1, 1]]
N = 1
NSP = (N + 1) * (N + 2) // 2
vanderm_matrix = np.zeros((NSP, NSP))
for rj in range(NSP):
    for i in range(N + 1):
        for j in range(N + 1):
            if i + j <= N:
                m = j + (N + 1) * i + 1 - i * (i - 1) // 2 - 1
                r = points_lists[rj][0]
                s = points_lists[rj][1]
                a, b = rstoab(r, s)
                test_p = simplex_2d_p(a, b, i, j)
                vanderm_matrix[rj][m] = test_p[0]
                print(f"rj = {rj}, m = {m},(i = {i}, j={j},) psi_m(r_j) = {test_p}")

print(f"vanderm_matrix={vanderm_matrix}")
print(f"inv = {np.linalg.inv(vanderm_matrix)}")

# plt.tight_layout()
# plt.show()
