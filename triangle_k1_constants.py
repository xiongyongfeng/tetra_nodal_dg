import numpy as np
from base.Vandermonde2D import vandermonde_2d
from base.GradVandermonde2D import grad_vandermonde_2d
from base.print_format import suppress_small
from base.Vandermonde2D import simplex_2d_p
from base.Dmatrices2D import dmatrices_2d
from base.Lift2D import lift_2d


def triangle_k1_constants():

    points_lists = [
        [-1, -1],
        [1, -1],
        [-1, 1],
    ]  # how to get lgl points of  higher order
    N = 1  # K1 element

    points_array = np.array(points_lists)
    r = points_array[:, 0]
    s = points_array[:, 1]

    V2D = vandermonde_2d(N, r, s)
    V2D_inv = np.linalg.inv(V2D)
    V2D_inv_T = V2D_inv.T
    print(f"V2D = {V2D}")
    print(f"modal_project_mat = {V2D_inv}")

    psi_0 = simplex_2d_p(r, s, 0, 0)
    Vi0 = V2D_inv_T[:, 0]
    weight = Vi0 * psi_0 * 2
    print(f"weight = {weight}")

    Mass_Matrix_inv = np.dot(V2D, V2D.T)
    Mass_Matrix = np.linalg.inv(Mass_Matrix_inv)  # Mass_Matrix = inv(V3D*V3D^T)

    print(f"Mass_Matrix_inv = {Mass_Matrix_inv}")
    print(f"Mass_Matrix = {Mass_Matrix}")

    # V2Dr, V2Ds = grad_vandermonde_2d(N, r, s)
    # print(f"V2Dr = {V2Dr}")
    # print(f"V2Ds = {V2Ds}")
    # Dr = np.dot(V2Dr, V2D_inv)
    # print(f"Dr = {Dr}")
    # Ds = np.dot(V2Ds, V2D_inv)
    # print(f"Ds = {Ds}")

    Dr, Ds = dmatrices_2d(N, r, s, V2D)
    print(f"Dr = {Dr}")
    print(f"Ds = {Ds}")

    Fmask = [[1, 2], [2, 0], [0, 1]]  # 定义了iface中ifp与isp的映射关系
    Nsp = (N + 1) * (N + 2) // 2  # 每个单元的点数（示例）
    Nfaces = 3  # 面数
    Nfp = N + 1  # 每个面的点数（示例）

    # 需要先定义 r, s, Fmask, V 等参数
    LIFT = lift_2d(N, Nsp, Nfaces, Nfp, r, s, Fmask, V2D)
    print(f"LIFT = {LIFT}")


if __name__ == "__main__":
    np.set_printoptions(
        formatter={"float": lambda x: f"{suppress_small(x):.23f}"}, suppress=True
    )
    triangle_k1_constants()
