from .matrix_utils import get_norm_laplacian, get_laplacian
import numpy as np
import scipy
from scipy.linalg import expm
import math


def renyi_divergence(A, B, q):
    if q <= 0:
        raise Exception("q must be >0")
    elif q == 1:
        answer = np.trace(
            np.dot(A, (scipy.linalg.logm(A) - scipy.linalg.logm(B)) / np.log(2.0))
        )
    else:
        answer = (
            (1 / (q - 1))
            * np.log(
                np.trace(
                    np.dot(
                        scipy.linalg.fractional_matrix_power(A, q),
                        scipy.linalg.fractional_matrix_power(B, 1 - q),
                    )
                )
            )
            / np.log(2.0)
        )
    return answer


def get_density_matrix(A, t, norm=0):
    A = A.astype(float)
    if norm:
        X = get_norm_laplacian(A)
    else:
        X = get_laplacian(A)

    R = expm(-t * X)
    R = R / np.trace(X)

    return R


def manual_entropy(pr):
    probs = np.trim_zeros(pr)
    probs = probs[np.where(probs > 1e-15)]
    return -np.real(np.sum(np.multiply(probs, np.log2(probs))))


def js_divergence(A, B, t, return_partial_entropies=True):
    X = get_density_matrix(A, t)
    Y = get_density_matrix(B, t)

    mixed = np.trim_zeros(np.linalg.eigvalsh((X + Y) / 2))
    raw1 = np.trim_zeros(np.linalg.eigvalsh(X))
    raw2 = np.trim_zeros(np.linalg.eigvalsh(Y))

    first = manual_entropy(mixed)
    ent1, ent2 = manual_entropy(raw1), manual_entropy(raw2)
    second = 0.5 * (ent1 + ent2)
    try:
        JSD = math.sqrt(first - second)
    except (ValueError, ArithmeticError):
        JSD = 0  # Handle negative values or other math errors

    if not return_partial_entropies:
        return JSD
    else:
        return manual_entropy(mixed), ent1, ent2, JSD
