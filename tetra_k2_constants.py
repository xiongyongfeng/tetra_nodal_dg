import numpy as np
from base.Vandermonde3D import vandermonde_3d
from base.Vandermonde3D import simplex_3d_p

from base.print_format import suppress_small
from base.Dmatrices3D import dmatrices_3d
from base.Lift3D import lift_3d
from base.SpaceInterpolateCoef import (
    space_interpolate_coef,
    grad_space_interpolate_coef,
)


def p2(x, y, z):
    return (
        x
        + 2 * y
        + 3 * z
        + 4 * x * x
        + 5 * y * y
        + 6 * z * z
        + 7 * x * y
        + 8 * x * z
        + 9 * y * z
    )


def dp2dx(x, y, z):
    return 1 + 8 * x + 7 * y + 8 * z


def dp2dy(x, y, z):
    return 2 + 10 * y + 7 * x + 9 * z


def dp2dz(x, y, z):
    return 3 + 12 * z + 8 * x + 9 * y


def tetra_k2_constants():
    points_lists = [
        [-1, -1, -1],  # point 0
        [1, -1, -1],  # point 1
        [-1, 1, -1],  # point 2
        [-1, -1, 1],  # point 3
        [0, -1, -1],  # point 4, midpoint of edge (0,1)
        [0, 0, -1],  # point 5, midpoint of edge (1,2)
        [-1, 0, -1],  # point 6, midpoint of edge (0,2)
        [-1, -1, 0],  # point 7, midpoint of edge (0,3)
        [0, -1, 0],  # point 8, midpoint of edge (1,3)
        [-1, 0, 0],  # point 9, midpoint of edge (2,3)
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

    C = space_interpolate_coef(N, r, s, t)
    print(f"space_interpolate_coef_matrix = \n{C}")

    Cr, Cs, Ct = grad_space_interpolate_coef(N, r, s, t)
    print(f"grad_interpolate_coef_matrix_r = \n{Cr}")
    print(f"grad_interpolate_coef_matrix_s = \n{Cs}")
    print(f"grad_interpolate_coef_matrix_t = \n{Ct}")

    Fmask = [
        [1, 2, 3, 5, 9, 8],
        [0, 3, 2, 7, 9, 6],
        [0, 1, 3, 4, 8, 7],
        [0, 2, 1, 6, 5, 4],
    ]  # 定义了iface中ifp与isp的映射关系
    Nsp = (N + 1) * (N + 2) * (N + 3) // 6  # 每个单元的点数（示例）
    Nfaces = 4  # 面数
    Nfp = (N + 1) * (N + 2) // 2  # 每个面的点数（示例）
    # # 需要先定义 r, s, Fmask, V 等参数
    LIFT = lift_3d(N, Nsp, Nfaces, Nfp, r, s, t, Fmask, V3D)
    print(f"LIFT =\n {LIFT}")

    u = p2(r, s, t)
    dudx = Dr @ u
    dudx_golden = dp2dx(r, s, t)
    print(f"dudx={dudx}")
    print(f"dudx_golden={dudx_golden}")


if __name__ == "__main__":
    # np.set_printoptions(precision=3)
    np.set_printoptions(
        formatter={"float": lambda x: f"{suppress_small(x):.2f}"}, suppress=True
    )
    tetra_k2_constants()
