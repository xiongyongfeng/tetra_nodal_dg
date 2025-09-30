import numpy as np
from base.JacobiGL import jacobi_gl
from base.Vandermonde1D import vandermonde_1d
from base.GradVandermonde1D import grad_vandermonde_1d


def line_segment_k1_constants():
    # 测试 JacobiGL 函数
    alpha, beta, N = 0, 0, 1  # 对应 Legendre 多项式
    x_gl = jacobi_gl(alpha, beta, N)

    V1D = vandermonde_1d(N, x_gl)
    print(f"VanderMat1D={V1D}")
    MassMat = np.linalg.inv(np.dot(V1D, V1D.T))
    print(f"MassMat={MassMat}")

    Vr = grad_vandermonde_1d(N, x_gl)
    Dr = np.dot(Vr, np.linalg.inv(V1D))
    print(f"Dr={Dr}")

    modal_project_mat = np.linalg.inv(V1D)
    print(f"modal_project_mat={modal_project_mat}")


if __name__ == "__main__":
    line_segment_k1_constants()
