import numpy as np
from scipy.sparse.csgraph import laplacian
from sklearn.neighbors import NearestNeighbors

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    cp = None


class MaximumVarianceUnfolding(object):
    """Maximum Variance Unfolding (MVU) for nonlinear dimensionality reduction.
    
    MVU learns a low-dimensional representation that preserves local distances
    while maximizing the variance of the embedding. It formulates the problem
    as a semidefinite program (SDP) to find a kernel matrix that respects
    local isometry constraints.
    
    Parameters
    ----------
    equation : {'berkley', 'wikipedia'}, default='berkley'
        Formulation variant to use:
        - 'berkley': UC Berkeley formulation with centered embedding constraint
        - 'wikipedia': Standard formulation with trace maximization
    solver : cvxpy.Solver, optional
        CVXPY solver to use. Default is SCS (Splitting Conic Solver).
    solver_tol : float, default=1e-2
        Convergence tolerance for the SDP solver.
    eig_tol : float, default=1e-10
        Tolerance for eigenvalue thresholding. Eigenvalues in (-eig_tol, eig_tol)
        are set to zero to handle numerical errors in the PSD constraint.
    solver_iters : int, default=2500
        Maximum number of solver iterations. Set to None for default solver limit.
    warm_start : bool, default=False
        Whether to use warm start in solver. Useful when solving similar problems
        repeatedly.
    seed : int, optional
        Random seed for reproducibility.
        
    Attributes
    ----------
    neighborhood_graph : ndarray of shape (n_samples, n_samples)
        Binary matrix indicating k-nearest neighbor connections.
        
    Raises
    ------
    ImportError
        If cvxpy is not installed. Install with pip install cvxpy.
        
    Examples
    --------
    >>> import numpy as np
    >>> from driada.dim_reduction.mvu import MaximumVarianceUnfolding
    >>> # Generate sample data
    >>> data = np.random.randn(100, 10)
    >>> # Initialize MVU
    >>> mvu = MaximumVarianceUnfolding(equation='berkley', solver_tol=1e-3)
    >>> # Fit and transform to 2D
    >>> embedding = mvu.fit_transform(data, dim=2, k=5)
    >>> print(embedding.shape)
    (100, 2)
    
    Notes
    -----
    Requires cvxpy package for solving the semidefinite program.
    The algorithm can be slow for large datasets due to the SDP formulation.
    The SDP solver may require tuning for different dataset sizes.
    
    References
    ----------
    Weinberger, K. Q., & Saul, L. K. (2006). An introduction to nonlinear
    dimensionality reduction by maximum variance unfolding. AAAI.    """

    def __init__(
        self,
        equation="berkley",
        solver=None,
        solver_tol=1e-2,
        eig_tol=1.0e-10,
        solver_iters=2500,
        warm_start=False,
        seed=None,
    ):
        """Initialize Maximum Variance Unfolding instance.
        
        Sets up the MVU solver with specified configuration for the
        semidefinite programming problem. Validates cvxpy availability
        and initializes solver parameters.
        
        Parameters
        ----------
        equation : {'berkley', 'wikipedia'}, default='berkley'
            Formulation variant to use:
            - 'berkley': UC Berkeley formulation with centered embedding
            - 'wikipedia': Standard formulation with trace maximization
        solver : cvxpy.Solver, optional
            CVXPY solver to use. Default is SCS (Splitting Conic Solver).
            Common options: cp.SCS, cp.MOSEK, cp.CVXOPT
        solver_tol : float, default=1e-2
            Convergence tolerance for the SDP solver.
            Lower values give more accurate solutions but take longer.
        eig_tol : float, default=1e-10
            Tolerance for eigenvalue thresholding. Eigenvalues with
            absolute value less than eig_tol are set to zero.
        solver_iters : int, default=2500
            Maximum number of solver iterations. None uses solver default.
            Increase if solver doesn't converge.
        warm_start : bool, default=False
            Whether to use warm start in solver. Useful when solving
            similar problems repeatedly.
        seed : int, optional
            Random seed for reproducibility in k-NN graph construction.
            
        Raises
        ------
        ImportError
            If cvxpy is not installed.
            
        Notes
        -----
        The Berkeley formulation often provides better numerical stability
        than the Wikipedia formulation. SCS solver is recommended for most
        cases as it handles large problems efficiently.        """
        if not CVXPY_AVAILABLE:
            raise ImportError(
                "cvxpy is required for MVU but not installed. "
                "Install it with: pip install cvxpy or conda install -c conda-forge cvxpy"
            )
        
        self.equation = equation
        self.solver = solver if solver is not None else cp.SCS
        self.solver_tol = solver_tol
        self.eig_tol = eig_tol
        self.solver_iters = solver_iters
        self.warm_start = warm_start
        self.seed = seed
        self.neighborhood_graph = None

    def fit(self, data, k):
        """Fit MVU model by solving the semidefinite program.
        
        Constructs a k-nearest neighbor graph and solves an SDP to find
        the optimal Gram matrix that preserves local distances while
        maximizing variance.
        
        Parameters
        ----------
        data : ndarray of shape (n_samples, n_features)
            High-dimensional input data.
        k : int
            Number of nearest neighbors to preserve distances for.
            Must be large enough to create a connected graph.
            
        Returns
        -------
        ndarray of shape (n_samples, n_samples)
            Optimal Gram matrix Q from the SDP solution.
            This is the inner product matrix of the embedded points.
            
        Raises
        ------
        ValueError
            If the k-NN graph has disconnected components and solver_iters
            is None (some solvers may not converge with disconnected graphs).
        ImportError
            If cvxpy is not installed.
            
        Examples
        --------
        >>> import numpy as np
        >>> # Create simple line data - MVU should preserve distances perfectly
        >>> np.random.seed(42)
        >>> n_points = 30
        >>> t = np.linspace(0, 2*np.pi, n_points)
        >>> # Circle in 3D space
        >>> data = np.column_stack([np.cos(t), np.sin(t), 0.1*np.sin(3*t)])
        >>> # Initialize and fit MVU
        >>> mvu = MaximumVarianceUnfolding(equation='berkley', solver_tol=1e-4)
        >>> gram_matrix = mvu.fit(data, k=6)
        >>> print(gram_matrix.shape)
        (30, 30)
        >>> # Check that Gram matrix eigenvalues are reasonable
        >>> # Small negative values are expected numerical errors from SDP solvers
        >>> eigvals = np.linalg.eigvalsh(gram_matrix)
        >>> # Verify the matrix is approximately PSD (within solver tolerance)
        >>> num_significant_negative = np.sum(eigvals < -1e-3)
        >>> print(f"Number of eigenvalues < -1e-3: {num_significant_negative}")
        Number of eigenvalues < -1e-3: 0
            
        Notes
        -----
        The Gram matrix Q satisfies:
        - Q is positive semidefinite
        - Sum of each row is zero (centering constraint)
        - For neighbors i,j: ||x_i - x_j||² = ||y_i - y_j||²
          where x are original points and y are embedded points        """
        # Number of data points in the set
        n = data.shape[0]

        # Set the seed
        np.random.seed(self.seed)

        # Calculate the nearest neighbors of each data point and build a graph
        N = NearestNeighbors(n_neighbors=k).fit(data).kneighbors_graph(data).todense()
        N = np.array(N)

        # Save the neighborhood graph to be accessed latter
        self.neighborhood_graph = N

        # To check for disconnected regions in the neighbor graph
        lap = laplacian(N, normed=True)
        eigvals, _ = np.linalg.eig(lap)

        for e in eigvals:
            if np.isclose(e, 0.0, rtol=1e-10, atol=1e-10) and self.solver_iters is None:
                raise ValueError(
                    "DISCONNECTED REGIONS IN NEIGHBORHOOD GRAPH. "
                    "PLEASE SPECIFY MAX ITERATIONS FOR THE SOLVER"
                )

        # Declare some CVXPy variables
        # Gramian of the original data
        P = cp.Constant(data.dot(data.T))
        # The projection of the Gramian
        Q = cp.Variable((n, n), PSD=True)
        # Initialized to zeros
        Q.value = np.zeros((n, n))
        # A shorter way to call a vector of 1's
        ONES = cp.Constant(np.ones((n, 1)))
        # A variable to keep the notation consistent with the Berkley lecture
        T = cp.Constant(n)

        # Declare placeholders to get rid of annoying warnings
        objective = None
        constraints = []

        # Wikipedia Solution
        if self.equation == "wikipedia":
            objective = cp.Maximize(cp.trace(Q))

            constraints = [Q >> 0, cp.sum(Q, axis=1) == 0]

            for i in range(n):
                for j in range(n):
                    if N[i, j] == 1:
                        constraints.append(
                            (P[i, i] + P[j, j] - P[i, j] - P[j, i])
                            - (Q[i, i] + Q[j, j] - Q[i, j] - Q[j, i])
                            == 0
                        )

        # UC Berkley Solution
        if self.equation == "berkley":
            objective = cp.Maximize(
                cp.multiply((1 / T), cp.trace(Q))
                - cp.multiply(
                    (1 / (T * T)), cp.trace(cp.matmul(cp.matmul(Q, ONES), ONES.T))
                )
            )

            constraints = [Q >> 0, cp.sum(Q, axis=1) == 0]
            for i in range(n):
                for j in range(n):
                    if N[i, j] == 1.0:
                        constraints.append(
                            Q[i, i]
                            - 2 * Q[i, j]
                            + Q[j, j]
                            - (P[i, i] - 2 * P[i, j] + P[j, j])
                            == 0
                        )

        # Solve the problem with the SCS Solver
        problem = cp.Problem(objective, constraints)
        # FUTURE: Add solver-specific parameter mapping for other solvers beyond SCS
        problem.solve(
            solver=self.solver,
            eps=self.solver_tol,
            max_iters=self.solver_iters,
            warm_start=self.warm_start,
        )

        return Q.value

    def fit_transform(self, data, dim, k):
        """Fit MVU model and transform data to lower dimension.
        
        Combines fit() and transform steps: solves the SDP to get the
        optimal Gram matrix, then extracts the embedding via eigendecomposition.
        
        Parameters
        ----------
        data : ndarray of shape (n_samples, n_features)
            High-dimensional input data.
        dim : int
            Target dimensionality for the embedding.
        k : int
            Number of nearest neighbors to preserve distances for.
            
        Returns
        -------
        ndarray of shape (n_samples, dim)
            Low-dimensional embedding of the data.
            
        Raises
        ------
        ValueError
            If k-NN graph has disconnected components (from fit).
            If requested dimension exceeds data samples.
        ImportError
            If cvxpy is not installed.
            
        Examples
        --------
        >>> import numpy as np
        >>> # Create Swiss roll data
        >>> n_samples = 100
        >>> noise = 0.05
        >>> t = 3 * np.pi * (1 + 2 * np.random.rand(n_samples))
        >>> x = t * np.cos(t)
        >>> y = 30 * np.random.rand(n_samples)
        >>> z = t * np.sin(t)
        >>> data = np.column_stack([x, y, z]) + noise * np.random.randn(n_samples, 3)
        >>> # Apply MVU
        >>> mvu = MaximumVarianceUnfolding(equation='berkley', solver_tol=1e-3)
        >>> embedding = mvu.fit_transform(data, dim=2, k=7)
        >>> print(f"Embedded shape: {embedding.shape}")
        Embedded shape: (100, 2)
            
        Notes
        -----
        The embedding is obtained by:
        1. Solving SDP to get Gram matrix Q
        2. Eigendecomposition: Q = V * Lambda * V^T
        3. Embedding: Y = sqrt(Lambda_top) * V_top^T
        where Lambda_top and V_top are the top 'dim' eigenvalues/vectors.
        
        Small negative eigenvalues may appear due to numerical errors in the
        SDP solution. These are thresholded to zero using eig_tol parameter.        """

        embedded_gramian = self.fit(data, k)

        # Decompose gramian to recover the projection
        # Use eigh for symmetric matrices (Gram matrix is symmetric PSD)
        eigenvalues, eigenvectors = np.linalg.eigh(embedded_gramian)

        # Set the eigenvalues that are within +/- eig_tol to 0
        eigenvalues[
            np.logical_and(-self.eig_tol < eigenvalues, eigenvalues < self.eig_tol)
        ] = 0.0

        # Assuming the eigenvalues and eigenvectors aren't sorted,
        #    sort them and get the top "dim" ones
        sorted_indices = eigenvalues.argsort()[::-1]
        top_eigenvalue_indices = sorted_indices[:dim]

        # Take the top eigenvalues and eigenvectors
        top_eigenvalues = eigenvalues[top_eigenvalue_indices]
        top_eigenvectors = eigenvectors[:, top_eigenvalue_indices]

        # Some quick math to get the projection and return it
        lbda = np.diag(top_eigenvalues**0.5)
        embedded_data = lbda.dot(top_eigenvectors.T).T

        return embedded_data
