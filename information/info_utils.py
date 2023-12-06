import numpy as np
from numba import njit

@njit()
def py_fast_digamma(data):
    res = np.zeros(len(data))
    for i, x in enumerate(data):
        "Faster digamma function assumes x > 0."
        r = 0
        while x <= 5:
            r -= 1 / x
            x += 1
        f = 1 / (x * x)
        t = f * (-1 / 12.0 + f * (1 / 120.0 + f * (-1 / 252.0 + f * (1 / 240.0 + f * (-1 / 132.0
                                                                                      + f * (691 / 32760.0 + f * (-1 / 12.0 + f * 3617 / 8160.0)))))))

        res[i] = r + np.log(x) - 0.5 / x + t

    return res