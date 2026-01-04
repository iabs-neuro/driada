from numpy import linalg as la
import numpy as np


def nearestPD(A):
    """Find the nearest positive-definite matrix to input.
    
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code,
    implementing Higham's algorithm for computing the nearest symmetric
    positive semidefinite matrix in the Frobenius norm.
    
    Parameters
    ----------
    A : array_like
        Input matrix, must be square. Can be non-symmetric.
        
    Returns
    -------
    numpy.ndarray
        The nearest positive-definite matrix to A.
        
    Notes
    -----
    The algorithm first symmetrizes the input matrix, then uses eigenvalue
    decomposition and iterative adjustment to ensure positive-definiteness.
    The function handles numerical precision issues that arise with matrices
    having eigenvalues near zero.
    
    References
    ----------
    .. [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    .. [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
           matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
           
    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[1, -1], [-1, 0]])  # Not positive-definite
    >>> A_pd = nearestPD(A)
    >>> eigvals = np.linalg.eigvals(A_pd)
    >>> np.all(eigvals >= 0)  # All eigenvalues are non-negative
    True
    >>> is_positive_definite(A_pd)  # Result is positive-definite
    True
    
    See Also
    --------
    ~driada.utils.matrix.is_positive_definite :
        Check if a matrix is positive-definite.
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if is_positive_definite(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # the order of 1e-16.
    I = np.eye(A.shape[0])
    k = 1
    while not is_positive_definite(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def is_positive_definite(B):
    """Returns true when input is positive-definite, via Cholesky.
    
    Tests whether a matrix is positive-definite by attempting Cholesky
    decomposition. A matrix is positive-definite if and only if it has
    a Cholesky decomposition.
    
    Parameters
    ----------
    B : array_like
        Square matrix to test for positive-definiteness.
        
    Returns
    -------
    bool
        True if the matrix is positive-definite, False otherwise.
        
    Notes
    -----
    This function uses the Cholesky decomposition as a numerical test
    for positive-definiteness. The Cholesky decomposition exists if and
    only if the matrix is symmetric (or Hermitian) and positive-definite.
    
    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[2, -1], [-1, 2]])  # Positive-definite
    >>> is_positive_definite(A)
    True
    
    >>> B = np.array([[1, 2], [2, 1]])  # Not positive-definite
    >>> is_positive_definite(B)
    False
    
    See Also
    --------
    ~driada.utils.matrix.nearestPD :
        Find the nearest positive-definite matrix.
    :func:`numpy.linalg.cholesky` :
        Cholesky decomposition.
    """
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False
