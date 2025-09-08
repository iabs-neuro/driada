import numpy as np
from ..utils.data import check_positive


def free_entropy(spectrum, t):
    """Calculate the free entropy from eigenvalue spectrum.

    Computes the free entropy (also known as free energy in physics) from
    a set of eigenvalues using the partition function approach.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Array of eigenvalues (e.g., from Laplacian or adjacency matrix).
    t : float
        Temperature-like parameter (inverse temperature in physics).

    Returns
    -------
    float
        Free entropy F = log2(Z), where Z is the partition function.

    Raises
    ------
    ValueError
        If spectrum is empty, if t is not positive, or if all exponentials 
        underflow to zero (partition function becomes zero).

    See Also
    --------
    ~driada.network.spectral.q_entropy : Rényi q-entropy from spectrum.
    ~driada.network.spectral.spectral_entropy : Von Neumann entropy from spectrum.

    Notes
    -----
    Based on equation 2 in https://www.nature.com/articles/s42005-021-00582-8#Sec10
    The partition function is Z = sum(exp(-t * λ_i)) where λ_i are eigenvalues.
    This measure is used in network thermodynamics and spectral analysis.

    Examples
    --------
    >>> spectrum = np.array([0, 1, 2, 3])
    >>> F = free_entropy(spectrum, t=1.0)
    >>> F > 0  # Free entropy is typically positive
    True    """
    # Input validation
    spectrum = np.asarray(spectrum)
    if spectrum.size == 0:
        raise ValueError("Spectrum array cannot be empty")
    check_positive(t=t)
    
    # eq.2 in https://www.nature.com/articles/s42005-021-00582-8#Sec10
    eigenvalues = np.exp(-t * spectrum)
    Z = np.sum(eigenvalues)
    
    # Check for underflow
    if Z <= 0 or not np.isfinite(Z):
        raise ValueError(f"Partition function underflow: Z = {Z}. "
                        "This can happen with large positive eigenvalues and large t.")
    
    F = np.log2(np.real(Z))
    return F


def q_entropy(spectrum, t, q=1):
    """Calculate the Rényi q-entropy from eigenvalue spectrum.

    Computes the Rényi entropy of order q for a density matrix derived
    from the eigenvalue spectrum, generalizing von Neumann entropy.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Array of eigenvalues λ_i (typically from graph Laplacian).
    t : float
        Inverse temperature parameter β.
    q : float, optional
        Order parameter for Rényi entropy. Default is 1 (von Neumann).
        Must be positive.

    Returns
    -------
    float
        Rényi q-entropy S_q(ρ) in bits (using log₂).

    Raises
    ------
    ValueError
        If spectrum is empty, if t is not positive, if q <= 0, or if 
        imaginary entropy is detected.

    See Also
    --------
    ~driada.network.spectral.free_entropy : Free entropy from spectrum.
    ~driada.network.spectral.spectral_entropy : Von Neumann entropy (q=1 case).

    Notes
    -----
    The Rényi q-entropy is defined as:
    - For q = 1: S_1(ρ) = -Tr(ρ log₂ ρ) (von Neumann entropy)
    - For q ≠ 1: S_q(ρ) = (1/(1-q)) log₂(Tr(ρ^q))
    
    For density matrix ρ = exp(-tL)/Z, this reduces to:
    S_q = (1/(1-q)) log₂(Z^(-q) sum_i exp(-tqλ_i))
    
    The Rényi entropy generalizes information measures:
    - q → 0: S_0 = log₂(rank(ρ)) (Hartley entropy)
    - q = 1: S_1 = von Neumann entropy
    - q → ∞: S_∞ = -log₂(λ_max) (min-entropy)

    References
    ----------
    De Domenico, M., & Biamonte, J. (2016). Spectral entropies as
    information-theoretic tools for complex network comparison.
    Physical Review X, 6(4), 041062.

    Examples
    --------
    >>> spectrum = np.array([0, 1, 2, 3])
    >>> S = q_entropy(spectrum, t=1.0, q=2)  # Rényi 2-entropy    """
    # Input validation
    spectrum = np.asarray(spectrum)
    if spectrum.size == 0:
        raise ValueError("Spectrum array cannot be empty")
    check_positive(t=t, q=q)
    
    Z = np.sum(np.exp(-t * spectrum))
    if q != 1:
        eigenvalues = np.exp(-t * q * spectrum)
        S = 1 / (1 - q) * np.log2(Z ** (-q) * np.sum(eigenvalues))
    else:
        S = spectral_entropy(spectrum, t, verbose=0)

    if np.imag(S) != 0:
        raise ValueError(f"Imaginary entropy detected: t={t}, q={q}, S={S}!")

    return S


def spectral_entropy(spectrum, t, verbose=0):
    """Calculate the von Neumann entropy from eigenvalue spectrum.

    Computes the von Neumann entropy S(ρ) = -Tr(ρ log₂ ρ) for a density
    matrix ρ constructed from the eigenvalue spectrum.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Array of eigenvalues λ_i (typically from graph Laplacian).
    t : float
        Inverse temperature parameter β (also interpreted as time).
    verbose : int, optional
        If > 0, print intermediate calculation steps. Default is 0.

    Returns
    -------
    float
        Von Neumann entropy S = -sum(p_i * log2(p_i)), where 
        p_i = exp(-t*λ_i) / Z and Z = sum(exp(-t*λ_i)).

    Raises
    ------
    ValueError
        If spectrum is empty, if t is not positive, or if partition function
        is zero (all probabilities underflow).

    See Also
    --------
    ~driada.network.spectral.q_entropy : Generalized Rényi entropy (reduces to this for q=1).
    ~driada.network.spectral.free_entropy : Related free energy measure.

    Notes
    -----
    For a density matrix ρ = exp(-tL)/Z derived from Laplacian L with
    eigenvalues λ_i, the von Neumann entropy reduces to:
    S(ρ) = -sum_i p_i log₂(p_i)
    where p_i = exp(-tλ_i) / Z are the diagonal elements in the
    eigenbasis of L.

    This entropy quantifies the "quantumness" or disorder in the network
    structure. Zero probabilities are automatically trimmed.

    References
    ----------
    De Domenico, M., & Biamonte, J. (2016). Spectral entropies as
    information-theoretic tools for complex network comparison.
    Physical Review X, 6(4), 041062.

    Examples
    --------
    >>> spectrum = np.array([0, 1, 1, 2])  # Laplacian eigenvalues
    >>> S = spectral_entropy(spectrum, t=1.0)
    >>> 0 <= S <= np.log2(len(spectrum))  # Entropy bounds
    True    """
    # Input validation
    spectrum = np.asarray(spectrum)
    if spectrum.size == 0:
        raise ValueError("Spectrum array cannot be empty")
    check_positive(t=t)
    
    eigenvalues = np.exp(-t * spectrum)
    Z = np.sum(eigenvalues)
    
    # Check for underflow
    if Z <= 0 or not np.isfinite(Z):
        raise ValueError(f"Partition function underflow: Z = {Z}. "
                        "This can happen with large positive eigenvalues and large t.")
    
    norm_eigenvalues = np.trim_zeros(eigenvalues / Z)
    
    # Handle edge case where all probabilities are trimmed
    if len(norm_eigenvalues) == 0:
        return 0.0
    
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
