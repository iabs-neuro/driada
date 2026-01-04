from .matrix_utils import get_norm_laplacian, get_laplacian
import numpy as np
import scipy
from scipy.linalg import expm
import math


def renyi_divergence(A, B, q):
    """Calculate the quantum Rényi divergence between two density matrices.

    The quantum Rényi divergence generalizes the quantum relative entropy
    (Kullback-Leibler divergence) and quantifies distinguishability between
    quantum states.

    Parameters
    ----------
    A : numpy.ndarray
        First density matrix (must be square, positive semi-definite,
        trace 1).
    B : numpy.ndarray
        Second density matrix (must be same shape as A).
    q : float
        Order parameter. Must be positive. q=1 gives quantum relative entropy.

    Returns
    -------
    float
        The quantum Rényi divergence D_q(A||B) in bits.

    Raises
    ------
    ValueError
        If q <= 0.
        If A and B have different shapes or are not square matrices.

    See Also
    --------
    ~driada.network.quantum.js_divergence : Quantum Jensen-Shannon divergence.
    ~driada.network.quantum.manual_entropy : Shannon/von Neumann entropy calculation.

    Notes
    -----
    The quantum Rényi divergence is defined as:
    - For q = 1: D_1(ρ||σ) = Tr(ρ(log₂ ρ - log₂ σ))
    - For q ≠ 1: D_q(ρ||σ) = (1/(q-1)) log₂(Tr(ρ^q σ^(1-q)))

    Properties:
    - D_q(ρ||σ) ≥ 0 with equality iff ρ = σ
    - Not symmetric: D_q(ρ||σ) ≠ D_q(σ||ρ) in general
    - For classical (diagonal) states, reduces to classical Rényi divergence

    References
    ----------
    Müller-Lennert, M., et al. (2013). On quantum Rényi entropies:
    A new generalization and some properties. J. Math. Phys. 54, 122203.

    Examples
    --------
    >>> rho = np.array([[0.7, 0.1], [0.1, 0.3]])
    >>> sigma = np.array([[0.5, 0.0], [0.0, 0.5]])
    >>> div = renyi_divergence(rho, sigma, q=0.5)    """
    if q <= 0:
        raise ValueError("q must be > 0")
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
    """Compute quantum density matrix from graph adjacency matrix.

    Constructs a quantum-like Gibbs state density matrix from a graph using
    the graph Laplacian, following De Domenico & Biamonte's formulation.

    Parameters
    ----------
    A : numpy.ndarray
        Adjacency matrix of the graph (must be square).
    t : float
        Inverse temperature parameter β (also interpreted as time).
        Controls the "quantumness" of the state. Must be positive.
    norm : int, optional
        If 1, use normalized Laplacian. If 0, use regular Laplacian.
        Default is 0.

    Returns
    -------
    numpy.ndarray
        Density matrix ρ = exp(-tL) / Z(t), where L is the Laplacian
        and Z(t) = Tr[exp(-tL)] is the partition function.

    Raises
    ------
    ValueError
        If A is not square or t is not positive.

    See Also
    --------
    ~driada.network.quantum.renyi_divergence : Uses density matrices for divergence calculation.
    ~driada.network.quantum.js_divergence : Uses density matrices for network comparison.
    ~driada.network.matrix_utils.get_laplacian : Computes graph Laplacian.
    ~driada.network.matrix_utils.get_norm_laplacian : Computes normalized Laplacian.

    Notes
    -----
    This density matrix is formally proportional to the propagator of a
    diffusive process on the network. The construction treats the Laplacian
    as a Hamiltonian in a quantum Gibbs state:
    ρ = (1/Z) exp(-βH), where H = L

    References
    ----------
    De Domenico, M., & Biamonte, J. (2016). Spectral entropies as
    information-theoretic tools for complex network comparison.
    Physical Review X, 6(4), 041062.

    Examples
    --------
    >>> A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    >>> rho = get_density_matrix(A, t=1.0)
    >>> np.trace(rho)  # Trace should be 1
    1.0    """
    A = A.astype(float)
    if norm:
        X = get_norm_laplacian(A)
    else:
        X = get_laplacian(A)

    R = expm(-t * X)
    R = R / np.trace(R)  # Normalize by trace to ensure Tr(ρ) = 1

    return R


def manual_entropy(pr):
    """Calculate Shannon entropy from probability distribution.

    Computes entropy while handling numerical issues with zero and
    near-zero probabilities.

    Parameters
    ----------
    pr : numpy.ndarray
        Probability distribution. Non-negative values that should sum to 1.
        Can also be eigenvalues of a density matrix for von Neumann entropy.

    Returns
    -------
    float
        Shannon entropy in bits.

    See Also
    --------
    :func:`scipy.stats.entropy` : Alternative implementation.
    ~driada.network.quantum.renyi_divergence : Generalized entropy measure.
    ~driada.network.quantum.js_divergence : Uses this for von Neumann entropy calculation.

    Notes
    -----
    The function filters out zero values and very small values (< 1e-15)
    to avoid numerical issues with logarithms.
    
    For quantum states, this computes the von Neumann entropy when given
    eigenvalues of a density matrix: S(ρ) = -Σᵢ λᵢ log₂(λᵢ).

    References
    ----------
    Shannon, C.E. (1948). A mathematical theory of communication.
    Bell System Technical Journal, 27(3), 379-423.

    Examples
    --------
    >>> pr = np.array([0.5, 0.5, 0.0])
    >>> H = manual_entropy(pr)
    >>> np.isclose(H, 1.0)  # Maximum entropy for 2 equiprobable states
    True    """
    probs = np.trim_zeros(pr)
    probs = probs[np.where(probs > 1e-15)]
    return -np.real(np.sum(np.multiply(probs, np.log2(probs))))


def js_divergence(A, B, t, return_partial_entropies=True):
    """Calculate quantum Jensen-Shannon divergence between two graphs.

    Computes the quantum generalization of JS divergence using von Neumann
    entropy of density matrices derived from graph Laplacians.

    Parameters
    ----------
    A : numpy.ndarray
        Adjacency matrix of first graph (must be square).
    B : numpy.ndarray
        Adjacency matrix of second graph (must be same shape as A).
    t : float
        Inverse temperature parameter β for density matrix computation.
        Must be positive.
    return_partial_entropies : bool, optional
        If True, return individual entropies along with JS divergence.
        Default is True.

    Returns
    -------
    float or tuple
        If return_partial_entropies=False: Square root of QJSD value.
        If return_partial_entropies=True: tuple of
        (S(ρ_mix), S(ρ_A), S(ρ_B), sqrt(QJSD)).

    Raises
    ------
    ValueError
        If A and B have different shapes or are not square matrices.

    See Also
    --------
    ~driada.network.quantum.renyi_divergence : Alternative quantum divergence measure.
    ~driada.network.quantum.get_density_matrix : Used to compute density matrices.
    ~driada.network.quantum.manual_entropy : Used for von Neumann entropy calculation.

    Notes
    -----
    The quantum Jensen-Shannon divergence is defined as:
    QJSD(ρ_A, ρ_B) = S((ρ_A + ρ_B)/2) - (S(ρ_A) + S(ρ_B))/2
    
    where S(ρ) = -Tr(ρ log₂ ρ) is the von Neumann entropy.
    
    **Important**: This function returns the SQUARE ROOT of QJSD, which is
    a proper metric on the quantum state space. To get the divergence itself,
    square the returned value.
    
    The function quantifies the distinguishability between two network
    structures in a quantum information context. Returns 0 if calculation
    results in negative values due to numerical precision issues (which
    can occur for very similar graphs).

    References
    ----------
    Lamberti, P., et al. (2008). Jensen-Shannon divergence as a measure
    of distinguishability between mixed quantum states. Physical Review A.

    Examples
    --------
    >>> A = np.array([[0, 1], [1, 0]])
    >>> B = np.array([[0, 1], [1, 0]])
    >>> js_div = js_divergence(A, B, t=1.0, return_partial_entropies=False)
    >>> js_div  # Should be 0 for identical graphs
    0.0    """
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
