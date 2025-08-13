import numpy as np


def free_entropy(spectrum, t):
    # eq.2 in https://www.nature.com/articles/s42005-021-00582-8#Sec10
    eigenvalues = np.exp(-t * spectrum)
    F = np.log2(np.real(np.sum(eigenvalues)))
    return F


def q_entropy(spectrum, t, q=1):
    """

    Args:
        spectrum:
        t:
        q:

    Returns:

    """
    # https://journals.aps.org/prx/abstract/10.1103/PhysRevX.6.041062

    if q <= 0:
        raise Exception("q must be >0")
    else:
        Z = np.sum(np.exp(-t * spectrum))
        if q != 1:
            eigenvalues = np.exp(-t * q * spectrum)
            S = 1 / (1 - q) * np.log(Z ** (-q) * np.sum(eigenvalues))
        else:
            S = spectral_entropy(spectrum, t, verbose=0)

    if np.imag(S) != 0:
        raise Exception(f"Imaginary entropy detected: t={t}, q={q}, S={S}!")

    return S


def spectral_entropy(spectrum, t, verbose=0):
    eigenvalues = np.exp(-t * spectrum)
    norm_eigenvalues = np.trim_zeros(eigenvalues / np.sum(eigenvalues))
    S = -np.real(np.sum(np.multiply(norm_eigenvalues, np.log2(norm_eigenvalues))))

    if verbose:
        print("initial eigenvalues:")
        print(spectrum)
        print("exp eigenvalues:")
        print(eigenvalues)
        print("norm exp eigenvalues:")
        print(norm_eigenvalues)
        print("logs:")
        print(np.log2(norm_eigenvalues))

    return S
