from .Vandermonde3D import vandermonde_3d
import numpy as np


def space_interpolate_coef(N, r, s, t):
    if N == 1:
        Mrst = np.array([[1, 1, 1, 1], r, s, t])
        C = np.linalg.inv(Mrst)
        return C
    if N == 2:
        rr = r * r
        ss = s * s
        tt = t * t
        rs = r * s
        rt = r * t
        st = s * t
        Mrst = np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], r, s, t, rr, ss, tt, rs, rt, st]
        )
        C = np.linalg.inv(Mrst)
        return C
