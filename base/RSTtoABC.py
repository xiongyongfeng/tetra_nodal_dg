import numpy as np


# 定义从 (r, s) 到 (a, b) 的转换函数
def rsttoabc(r, s, t):
    """
    Transfer from (r, s, t) coordinates to (a, b, c) coordinates in a triangle.

    Parameters:
    r, s, t : array_like
        Input coordinates. Can be scalars or arrays.

    Returns:
    a, b, c : ndarray or scalars
        Transformed coordinates. Returns scalars if all inputs are scalars.
    """
    r_arr = np.asarray(r)
    s_arr = np.asarray(s)
    t_arr = np.asarray(t)

    # 使用 errstate 上下文管理器忽略除零和无效操作的警告
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator_st = -s_arr - t_arr
        a_arr = np.where(denominator_st != 0, 2 * (1 + r_arr) / denominator_st - 1, -1)

        denominator_t = 1 - t_arr
        b_arr = np.where(denominator_t != 0, 2 * (1 + s_arr) / denominator_t - 1, -1)

    c_arr = t_arr

    # 如果原始输入都是标量，则返回标量
    if np.isscalar(r) and np.isscalar(s) and np.isscalar(t):
        # 使用 .item() 替代已移除的 np.asscalar
        return a_arr.item(), b_arr.item(), c_arr.item()
    else:
        return a_arr, b_arr, c_arr
