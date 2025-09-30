def suppress_small(x):
    """将绝对值小于1e-5的值格式化为0,其他值保留原样"""
    if abs(x) < 1e-10:
        return 0.0
    else:
        return x
