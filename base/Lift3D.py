import numpy as np
from .Vandermonde2D import vandermonde_2d


def lift_3d(N, Np, Nfaces, Nfp, r, s, t, Fmask, V):
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

    for iface in range(0, 4):
        indices = Fmask[iface]
        if iface == 0:
            faceR = [r[i] for i in indices]
            faceS = [s[i] for i in indices]
        elif iface == 1:
            faceR = [s[i] for i in indices]
            faceS = [t[i] for i in indices]
        elif iface == 2:
            faceR = [r[i] for i in indices]
            faceS = [t[i] for i in indices]
        elif iface == 3:
            faceR = [r[i] for i in indices]
            faceS = [s[i] for i in indices]
        V2D = vandermonde_2d(N, faceR, faceS)
        mass_iface = np.linalg.inv(V2D @ V2D.T)
        if iface == 0:
            mass_iface = mass_iface * np.sqrt(3.0)
        Emat[indices, iface * Nfp : (iface + 1) * Nfp] = mass_iface

    # inv(mass matrix)*\I_n(L_i,L_j)_{edge_n}
    # LIFT = V @ (V.T @ Emat)
    LIFT = Emat

    return LIFT
