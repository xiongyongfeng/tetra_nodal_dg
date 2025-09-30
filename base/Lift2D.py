import numpy as np
from .Vandermonde1D import vandermonde_1d


def lift_2d(N, Np, Nfaces, Nfp, r, s, Fmask, V):
    """
    Compute surface to volume lift term for DG formulation

    Parameters:
    N: 多项式阶数
    Np: 每个单元的解点数
    Nfaces: 面数
    Nfp: 每个面的点数
    r, s: 参考坐标
    Fmask: 面掩码数组
    V: 范德蒙矩阵

    Returns:
    LIFT: 提升矩阵
    """

    Emat = np.zeros((Np, Nfaces * Nfp))

    for iface in range(0, 3):
        indices = Fmask[iface]
        if iface == 0:
            faceR = [r[i] for i in indices]
        elif iface == 1:
            faceR = [s[i] for i in indices]
        elif iface == 2:
            faceR = [r[i] for i in indices]
        V1D = vandermonde_1d(N, faceR)
        mass_iface = np.linalg.inv(V1D @ V1D.T)
        if iface == 0:
            mass_iface = mass_iface * np.sqrt(2)
        Emat[indices, iface * Nfp : (iface + 1) * Nfp] = mass_iface

    # inv(mass matrix)*\I_n(L_i,L_j)_{edge_n}
    # LIFT = V @ (V.T @ Emat)
    LIFT = Emat

    return LIFT
