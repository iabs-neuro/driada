import numpy as np

def free_entropy(spectrum, t):
    eigenvalues = np.exp(-t * spectrum)
    F = np.log2(np.real(np.sum(eigenvalues)))
    return F


def spectral_entropy(spectrum, t, verbose=0):
    eigenvalues = np.exp(-t * spectrum)
    norm_eigenvalues = np.trim_zeros(eigenvalues / np.sum(eigenvalues))

    if verbose:
        print('initial eigenvalues:')
        print(spectrum)
        print('exp eigenvalues:')
        print(eigenvalues)
        print('norm exp eigenvalues:')
        print(norm_eigenvalues)
        print('logs:')
        print(np.log2(norm_eigenvalues))

    # https://journals.aps.org/prx/abstract/10.1103/PhysRevX.6.041062

    S = -np.real(np.sum(np.multiply(norm_eigenvalues, np.log2(norm_eigenvalues))))
    return S