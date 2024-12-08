import numpy as np
from numba import njit

@njit()
def py_fast_digamma_arr(data):
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


@njit()
def py_fast_digamma(x):
    r = 0
    x = x*1.0
    while x <= 5:
        r -= 1 / x
        x += 1
    f = 1 / (x * x)
    t = f * (-1 / 12.0 + f * (1 / 120.0 + f * (-1 / 252.0 + f * (1 / 240.0 + f * (-1 / 132.0
                                                                                  + f * (691 / 32760.0 + f * (-1 / 12.0 + f * 3617 / 8160.0)))))))

    res = r + np.log(x) - 0.5 / x + t
    return res


def binary_mi_score(contingency):
    nzx, nzy = np.nonzero(contingency)
    nz_val = contingency[nzx, nzy]

    contingency_sum = contingency.sum()
    pi = np.ravel(contingency.sum(axis=1))
    pj = np.ravel(contingency.sum(axis=0))

    # Since MI <= min(H(X), H(Y)), any labelling with zero entropy, i.e. containing a
    # single cluster, implies MI = 0
    if pi.size == 1 or pj.size == 1:
        return 0.0

    log_contingency_nm = np.log(nz_val)
    contingency_nm = nz_val / contingency_sum
    # Don't need to calculate the full outer product, just for non-zeroes
    outer = pi.take(nzx).astype(np.int64, copy=False) * pj.take(nzy).astype(
        np.int64, copy=False
    )
    log_outer = -np.log(outer) + np.log(pi.sum()) + np.log(pj.sum())
    mi = (
            contingency_nm * (log_contingency_nm - np.log(contingency_sum))
            + contingency_nm * log_outer
    )
    mi = np.where(np.abs(mi) < np.finfo(mi.dtype).eps, 0.0, mi)
    return np.clip(mi.sum(), 0.0, None)