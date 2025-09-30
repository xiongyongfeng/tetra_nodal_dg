import numpy as np
from base.Vandermonde3D import vandermonde_3d
from base.Vandermonde3D import simplex_3d_p

from base.print_format import suppress_small
from base.Dmatrices3D import dmatrices_3d
from base.Lift3D import lift_3d


def tetra_k2_constants():
    points_lists = [
        [-1, -1, -1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [0, -1, -1],
        [0, 0, -1],
        [-1, 0, -1],
        [0, -1, 0],
        [-1, 0, 0],
        [-1, -1, 0],
    ]  # how to get lgl points of  higher order
    N = 2  # K2 element

    points_array = np.array(points_lists)
    r = points_array[:, 0]
    s = points_array[:, 1]
    t = points_array[:, 2]

    V3D = vandermonde_3d(N, r, s, t)
    V3D_inv = np.linalg.inv(V3D)
    V3D_inv_T = V3D_inv.T
    print(f"Vandermonde Matrix =\n {V3D}")
    print(f"modal_project_mat(V_inv) =\n {V3D_inv}")

    psi_0 = simplex_3d_p(r, s, t, 0, 0, 0)
    Vi0 = V3D_inv_T[:, 0]
    weight = Vi0 * psi_0 * 4 / 3
    print(f"weight =\n {weight}")

    Mass_Matrix_inv = np.dot(V3D, V3D.T)
    Mass_Matrix = np.linalg.inv(Mass_Matrix_inv)  # Mass_Matrix = inv(V3D*V3D^T)

    print(f"Mass_Matrix =\n {Mass_Matrix}")
    print(f"Mass_Matrix_inv =\n {Mass_Matrix_inv}")

    # V3Dr, V3Ds, V3Dt = grad_vandermonde_3d(N, r, s, t)
    # Dr = np.dot(V3Dr, V3D_inv)
    # print(f"Dr = {Dr}")
    # Ds = np.dot(V3Ds, V3D_inv)
    # print(f"Ds = {Ds}")
    # Dt = np.dot(V3Dt, V3D_inv)
    # print(f"Dt = {Dt}")
    Dr, Ds, Dt = dmatrices_3d(N, r, s, t, V3D)
    print(f"Dr =\n {Dr}")
    print(f"Ds =\n {Ds}")
    print(f"Dt =\n {Dt}")

    # Fmask = [
    #     [1, 2, 3],
    #     [0, 3, 2],
    #     [0, 1, 3],
    #     [0, 2, 1],
    # ]  # 定义了iface中ifp与isp的映射关系
    # Nsp = (N + 1) * (N + 2) * (N + 3) // 6  # 每个单元的点数（示例）
    # Nfaces = 4  # 面数
    # Nfp = (N + 1) * (N + 2) // 2  # 每个面的点数（示例）
    # # 需要先定义 r, s, Fmask, V 等参数
    # LIFT = lift_3d(N, Nsp, Nfaces, Nfp, r, s, t, Fmask, V3D)
    # print(f"LIFT =\n {LIFT}")


if __name__ == "__main__":
    # np.set_printoptions(precision=3)
    np.set_printoptions(
        formatter={"float": lambda x: f"{suppress_small(x):.2f}"}, suppress=True
    )
    tetra_k2_constants()
