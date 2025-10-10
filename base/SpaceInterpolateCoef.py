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


def grad_space_interpolate_coef(N, r, s, t):
    interp_coef = space_interpolate_coef(N, r, s, t)
    grad_interp_coef_r = np.zeros_like(interp_coef)
    grad_interp_coef_s = np.zeros_like(interp_coef)
    grad_interp_coef_t = np.zeros_like(interp_coef)
    if N == 1:
        grad_interp_coef_r[:, 0] = interp_coef[:, 1]
        grad_interp_coef_s[:, 0] = interp_coef[:, 2]
        grad_interp_coef_t[:, 0] = interp_coef[:, 3]

    if N == 2:  # 1, r, s, t, rr, ss, tt, rs, rt, st
        grad_interp_coef_r[:, 0] = interp_coef[:, 1]
        grad_interp_coef_r[:, 1] = interp_coef[:, 4] * 2.0
        grad_interp_coef_r[:, 2] = interp_coef[:, 7]
        grad_interp_coef_r[:, 3] = interp_coef[:, 8]

        grad_interp_coef_s[:, 0] = interp_coef[:, 2]
        grad_interp_coef_s[:, 1] = interp_coef[:, 7]
        grad_interp_coef_s[:, 2] = interp_coef[:, 5] * 2.0
        grad_interp_coef_s[:, 3] = interp_coef[:, 9]

        grad_interp_coef_t[:, 0] = interp_coef[:, 3]
        grad_interp_coef_t[:, 1] = interp_coef[:, 8]
        grad_interp_coef_t[:, 2] = interp_coef[:, 9]
        grad_interp_coef_t[:, 3] = interp_coef[:, 6] * 2

    return grad_interp_coef_r, grad_interp_coef_s, grad_interp_coef_t
