import numpy as np


# 定义从 (r, s) 到 (a, b) 的转换函数
def rstoab(r, s):
    """
    将 (r, s) 坐标转换到 (a, b) 坐标
    公式: a = 2*(1+r)/(1-s) - 1, b = s
    """
    r = np.asarray(r)
    s = np.asarray(s)

    Np = r.size
    # 关键修改：将数组 a 初始化为浮点类型，例如 float64
    a = np.zeros(Np, dtype=np.float64)  # 初始化浮点数组
    b = np.zeros(Np, dtype=np.float64)  # 通常b也应为浮点，除非你确定s永远是整数

    # 使用 np.divide 并指定输出到浮点数组 a
    # 同时处理 s != 1 的条件
    np.divide(2 * (1 + r), (1 - s), out=a, where=s != 1)
    # 对于 where 条件为 False 的位置（即 s == 1），上一步没有写入，它们仍然是初始值0。所以需要额外处理这些位置。
    a[s == 1] = 1  # 或者根据你的需求设置为其他值，例如 np.inf
    a[s != 1] -= 1  # 对所有计算过的值减去1

    b = s  # 直接赋值
    return a, b
