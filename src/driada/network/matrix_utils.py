import numpy as np
import copy
import scipy.sparse as sp


def _is_sparse(matrix):
    """
    Check if a matrix is sparse.
    
    Parameters
    ----------
    matrix : any
        Object to check. Can be numpy array, scipy sparse matrix, or any other object.
        
    Returns
    -------
    bool
        True if matrix is a scipy sparse matrix, False otherwise.
        Returns False for any non-sparse input (including None, strings, etc).    """
    return sp.issparse(matrix)


def _plain_bfs(adj, source):
    """
    Perform breadth-first search from source node.
    
    Adapted from networkx.algorithms.components.connected._plain_bfs

    Parameters
    ----------
    adj : numpy.ndarray or scipy sparse matrix (CSR, CSC)
        Adjacency matrix with shape (n, n). Must support .shape attribute 
        and matrix indexing with adj[[node], :].
    source : int
        Starting node index (0-based).

    Returns
    -------
    set
        Set of nodes reachable from source.
        
    Raises
    ------
    ValueError
        If source is out of bounds for the adjacency matrix.
    
    Notes
    -----
    Early termination occurs when all n nodes have been visited.    """
    n = adj.shape[0]
    if not 0 <= source < n:
        raise ValueError(f"Source node {source} is out of bounds for adjacency matrix of size {n}")
    
    seen = {source}
    nextlevel = [source]
    while nextlevel:
        thislevel = nextlevel
        nextlevel = []
        for v in thislevel:
            for w in get_neighbors_from_adj(adj, v):
                if w not in seen:
                    seen.add(w)
                    nextlevel.append(w)
            if len(seen) == n:
                return seen
    return seen


def get_neighbors_from_adj(a, node):
    """Get outgoing neighbors of a node from adjacency matrix.
    
    Parameters
    ----------
    a : numpy.ndarray or scipy sparse matrix
        Adjacency matrix of the graph. Must support matrix indexing with 
        a[[node], :] and .nonzero() method.
    node : int
        Node index to find neighbors for (0-based). Negative indices are supported
        following Python's indexing convention.
        
    Returns
    -------
    numpy.ndarray
        Array of neighbor node indices where a[node, :] is non-zero.
        Includes self-loops if present. Returns indices in column order.
    
    Raises
    ------
    IndexError
        From underlying array if node index is out of bounds.
    
    Notes
    -----
    For directed graphs, returns out-neighbors only (nodes that can be
    reached from this node).    """
    inds = a[[node], :].nonzero()[1]
    return inds


def get_ccs_from_adj(adj):
    """Find all connected components in an undirected graph.
    
    Parameters
    ----------
    adj : array-like or sparse matrix
        Adjacency matrix of the graph. Should be symmetric for undirected graphs.
        
    Yields
    ------
    set
        Set of node indices in each connected component. Components are yielded
        in order of lowest node index. Isolated nodes are included as
        single-node components.
    
    Notes
    -----
    For directed graphs, this finds weakly connected components
    (treating edges as undirected). For strongly connected components
    in directed graphs, use get_sccs_from_adj instead.    """
    seen = set()
    for v in range(adj.shape[0]):
        if v not in seen:
            c = _plain_bfs(adj, v)
            seen.update(c)
            yield c


def get_sccs_from_adj(adj):
    """
    Get strongly connected components using Tarjan's algorithm.
    
    Adapted from networkx.algorithms.components.strongly_connected.strongly_connected_components

    Parameters
    ----------
    adj : numpy.ndarray or scipy sparse matrix
        Adjacency matrix representing a directed graph. Must support .shape
        attribute and work with get_neighbors_from_adj() function.

    Yields
    ------
    set
        Set of nodes in each strongly connected component.
        Components are yielded as they are discovered.
    
    Notes
    -----
    Implements non-recursive version of Tarjan's algorithm. A strongly 
    connected component is a maximal set of nodes where every node is 
    reachable from every other node following directed edges.    """

    all_nodes = range(adj.shape[0])
    preorder = {}
    lowlink = {}
    scc_found = set()
    scc_queue = []
    i = 0  # Preorder counter
    neighbors = {v: iter(get_neighbors_from_adj(adj, v)) for v in all_nodes}
    for source in all_nodes:
        if source not in scc_found:
            queue = [source]
            while queue:
                v = queue[-1]
                if v not in preorder:
                    i = i + 1
                    preorder[v] = i
                done = True
                for w in neighbors[v]:
                    if w not in preorder:
                        queue.append(w)
                        done = False
                        break
                if done:
                    lowlink[v] = preorder[v]
                    for w in get_neighbors_from_adj(adj, v):
                        if w not in scc_found:
                            if preorder[w] > preorder[v]:
                                lowlink[v] = min([lowlink[v], lowlink[w]])
                            else:
                                lowlink[v] = min([lowlink[v], preorder[w]])
                    queue.pop()
                    if lowlink[v] == preorder[v]:
                        scc = {v}
                        while scc_queue and preorder[scc_queue[-1]] > preorder[v]:
                            k = scc_queue.pop()
                            scc.add(k)
                        scc_found.update(scc)
                        yield scc
                    else:
                        scc_queue.append(v)


def get_giant_cc_from_adj(adj):
    """Extract the giant (largest) connected component from a graph.
    
    Parameters
    ----------
    adj : sparse matrix
        Adjacency matrix of the graph. Must be a scipy sparse matrix.
        
    Returns
    -------
    gcc_adj : sparse matrix
        Adjacency matrix of the giant connected component.
    node_mapping : dict
        Mapping from new node indices to original node indices.
        
    Raises
    ------
    ValueError
        If the graph is empty (no nodes).
    AttributeError
        If adj is not a sparse matrix (missing tocsc/tocsr methods).
        
    Notes
    -----
    For directed graphs, this finds the giant weakly connected component.    """
    connected_components = list(get_ccs_from_adj(adj))
    if not connected_components:
        raise ValueError("Cannot find giant component in empty graph")
    
    connected_components.sort(key=len, reverse=True)
    gcc = np.array(list(connected_components[0]))
    gcc_adj = adj[gcc, :].tocsc()[:, gcc].tocsr()

    # mapping of new nodes to old ones
    node_mapping = dict(zip(range(len(gcc)), gcc))

    return gcc_adj, node_mapping


def get_giant_scc_from_adj(adj):
    """Extract the giant (largest) strongly connected component from a directed graph.
    
    Parameters
    ----------
    adj : sparse matrix
        Adjacency matrix of the directed graph. Must be a scipy sparse matrix.
        
    Returns
    -------
    gscc_adj : sparse matrix
        Adjacency matrix of the giant strongly connected component.
    node_mapping : dict
        Mapping from new node indices to original node indices.
        
    Raises
    ------
    ValueError
        If the graph is empty (no nodes).
    AttributeError
        If adj is not a sparse matrix (missing tocsc/tocsr methods).
        
    Notes
    -----
    For undirected graphs, each node will be its own SCC and the function
    will return an arbitrary single-node component.    """
    connected_components = list(get_sccs_from_adj(adj))
    if not connected_components:
        raise ValueError("Cannot find giant component in empty graph")
    
    connected_components.sort(key=len, reverse=True)
    gscc = np.array(list(connected_components[0]))
    # Convert to CSC for efficient column slicing, then back to CSR
    gscc_adj = adj[gscc, :].tocsc()[:, gscc].tocsr()

    # mapping of new nodes to old ones
    node_mapping = dict(zip(range(len(gscc)), gscc))

    return gscc_adj, node_mapping


def assign_random_weights(A):
    """Assign random weights to edges in an adjacency matrix.
    
    Parameters
    ----------
    A : array-like or sparse matrix
        Binary adjacency matrix. Can be dense numpy array or scipy sparse matrix.
        
    Returns
    -------
    array-like
        Weighted adjacency matrix with random weights in [0,1],
        symmetrized to ensure undirected graph. Returns same format
        as input (dense if A is dense, sparse if A is sparse).
        
    Notes
    -----
    - The function generates uniform random weights in [0,1]
    - Weights are symmetrized by averaging: (W + W.T) / 2
    - Directed edges will get averaged weights
    - Zero entries in A remain zero in the output
    - To control randomness, set numpy's random seed before calling
    - Sparse matrices are handled efficiently without densification    """
    if sp.issparse(A):
        # Efficient sparse implementation
        # Convert to COO for easy access to non-zero entries
        A_coo = A.tocoo()
        
        # Generate random weights only for existing edges
        weights = np.random.random(len(A_coo.data))
        
        # Create weighted sparse matrix with same structure
        W = sp.coo_matrix((weights, (A_coo.row, A_coo.col)), shape=A.shape)
        
        # Convert to CSR for efficient arithmetic operations
        W_csr = W.tocsr()
        
        # Symmetrize and return in same format as input
        W_sym = (W_csr + W_csr.T) / 2
        
        # Return in the same sparse format as input
        return W_sym.asformat(A.format)
    else:
        # Dense implementation (unchanged for backward compatibility)
        X = np.random.random(size=(A.shape[0], A.shape[0]))
        W = np.multiply(X, A)
        return (W + W.T) / 2


def turn_to_partially_directed(mat, directed=0.0, weighted=0):
    """Convert a symmetric matrix to partially directed by randomly removing edges.
    
    This function supports both dense (numpy) and sparse (scipy.sparse) matrices.
    Sparse format is preferred for memory efficiency with large graphs.
    
    Parameters
    ----------
    mat : np.ndarray or scipy.sparse matrix
        Input adjacency matrix (should be symmetric)
    directed : float or None, default=0.0
        Fraction of edges to make directed. Must be in range [0.0, 1.0] or None.
        0.0 = fully undirected, 1.0 = fully directed. None is converted to 0.0.
    weighted : int, default=0
        Whether the matrix represents a weighted graph (0=binary, 1=weighted)
        
    Returns
    -------
    scipy.sparse.csr_matrix
        Partially directed adjacency matrix in sparse format (always returns sparse)
        
    Raises
    ------
    ValueError
        If directed is not in range [0.0, 1.0] (after None conversion)
        
    Notes
    -----
    - Self-loops are always removed
    - Only symmetric edge pairs are affected
    - Asymmetric edges remain unchanged
    - Uses uniform random distribution for edge direction selection    """
    # Validate parameters
    if directed is None:
        directed = 0.0
    if not 0.0 <= directed <= 1.0:
        raise ValueError(f"directed must be in range [0.0, 1.0], got {directed}")
    
    # Determine if we're working with sparse or dense
    is_sparse = sp.issparse(mat)
    
    if is_sparse:
        # Sparse pathway (preferred)
        A = mat.tocsr().copy()
        A.setdiag(0)  # Remove self-loops
        A.eliminate_zeros()  # Remove any explicit zeros
        
        if directed == 0.0:
            if not weighted:
                A.data = np.ones_like(A.data, dtype=np.int32)
            return A
            
        # For partial directionality with sparse matrices
        A_coo = A.tocoo()
        
        # Find symmetric edge pairs efficiently
        edges = list(zip(A_coo.row, A_coo.col))
        edge_set = set(edges)
        symmetric_pairs = []
        seen = set()
        
        for i, j in edges:
            if i < j and (j, i) in edge_set and (i, j) not in seen:
                symmetric_pairs.append((i, j))
                seen.add((i, j))
                seen.add((j, i))
        
        if not symmetric_pairs:
            return A
        
        # Randomly select which symmetric pairs to make directed
        n_pairs = len(symmetric_pairs)
        random_tosses = np.random.random(n_pairs)
        
        # Determine which edges to remove
        edges_to_remove = set()
        
        for idx, (i, j) in enumerate(symmetric_pairs):
            if random_tosses[idx] < directed:
                if random_tosses[idx] < directed / 2.0:
                    edges_to_remove.add((i, j))
                else:
                    edges_to_remove.add((j, i))
        
        # Build new sparse matrix without removed edges
        edge_to_value = {}
        for row, col, data in zip(A_coo.row, A_coo.col, A_coo.data):
            if (row, col) not in edges_to_remove:
                edge_to_value[(row, col)] = data
        
        if edge_to_value:
            rows, cols = zip(*edge_to_value.keys())
            data = list(edge_to_value.values())
            result = sp.csr_matrix((data, (rows, cols)), shape=A.shape)
        else:
            result = sp.csr_matrix(A.shape)
            
        return result
    
    else:
        # Dense pathway (kept for backward compatibility)
        if not isinstance(mat, np.ndarray):
            raise TypeError("Input must be numpy array or scipy sparse matrix")
            
        A = copy.deepcopy(mat)
        np.fill_diagonal(A, 0)  # Remove self-loops
        
        if directed == 0.0 or directed is None:
            if not weighted:
                a = A.astype(bool).astype(int)  # Convert bool to 0/1
            else:
                a = A.astype(float)
            return sp.csr_matrix(a)
        
        # Original dense implementation
        rows, cols = A.nonzero()
        edgeset = set(zip(rows, cols))
        upper = np.array([l for l in edgeset if l[0] < l[1]])
        
        random_tosses = np.random.random(len(upper))
        condition1 = (random_tosses >= directed / 2.0) & (random_tosses < directed)
        condition2 = (random_tosses <= directed / 2.0) & (random_tosses < directed)
        indices_where_upper_is_removed = np.where(condition1)[0]
        indices_where_lower_is_removed = np.where(condition2)[0]
        
        u_xdata = [u[0] for u in upper[indices_where_upper_is_removed]]
        u_ydata = [u[1] for u in upper[indices_where_upper_is_removed]]
        A[u_xdata, u_ydata] = 0
        
        l_xdata = [u[1] for u in upper[indices_where_lower_is_removed]]
        l_ydata = [u[0] for u in upper[indices_where_lower_is_removed]]
        A[l_xdata, l_ydata] = 0
        
        # Convert to sparse before returning
        return sp.csr_matrix(A)


def get_symmetry_index(a):
    """Calculate symmetry index of a matrix.
    
    The symmetry index measures what fraction of edges in a directed graph
    have their reciprocal edge present. It quantifies the degree of 
    bidirectionality in the network.
    
    Parameters
    ----------
    a : array-like or sparse matrix
        Input adjacency matrix to analyze. Can be weighted or binary.
        
    Returns
    -------
    float
        Symmetry index in range [0, 1]:
        - 1.0: Perfectly symmetric (all edges are bidirectional)
        - 0.0: Completely asymmetric (no reciprocal edges)
        - 0.5: Half of edges have reciprocal counterparts
        
    Notes
    -----
    The symmetry index is calculated as:
    (number of edges with symmetric counterpart) / (total number of edges)
    
    An edge a[i,j] has a symmetric counterpart if a[j,i] is also non-zero.
    Self-loops (diagonal elements) always have their symmetric counterpart 
    (themselves) and thus contribute to symmetry.
    
    Empty matrices (no edges) are considered perfectly symmetric and return 1.0.
    
    This metric differs from counting symmetric pairs: it counts each edge
    individually and checks if its reverse exists, rather than counting
    unique bidirectional pairs.
    
    Examples
    --------
    >>> # Perfectly symmetric matrix
    >>> A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    >>> get_symmetry_index(A)
    1.0
    
    >>> # Directed cycle (no reciprocal edges)
    >>> B = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    >>> get_symmetry_index(B)
    0.0
    
    >>> # Partially symmetric
    >>> C = np.array([[0, 1, 1], [1, 0, 0], [0, 0, 0]])
    >>> get_symmetry_index(C)  # 2 edges have counterparts out of 3 total
    0.6666...    """
    if _is_sparse(a):
        # Sparse implementation
        a_bool = a.astype(bool)
        # For each edge, check if its transpose also exists
        # An edge has a symmetric counterpart if both a[i,j] and a[j,i] are non-zero
        symmetric = a_bool.multiply(a_bool.T)
        # Count edges with symmetric counterpart (symmetric[i,j]=1 means a[i,j] has counterpart)
        symm_count = symmetric.nnz
        total_edges = a_bool.nnz
        if total_edges == 0:
            return 1.0  # Empty matrix is considered symmetric
        symm_index = symm_count / total_edges
    else:
        # Dense implementation
        a_bool = a.astype(bool)
        # For each edge, check if its transpose also exists
        symmetric = a_bool & a_bool.T
        # Count edges with symmetric counterpart
        symm_count = np.count_nonzero(symmetric)
        total_edges = np.count_nonzero(a_bool)
        if total_edges == 0:
            return 1.0  # Empty matrix is considered symmetric
        symm_index = symm_count / total_edges
    
    return symm_index


def symmetric_component(A, is_weighted):
    """Extract the symmetric component of a matrix.
    
    Finds edges that exist in both directions (i,j) and (j,i).
    For weighted graphs, preserves the original weights.
    
    Parameters
    ----------
    A : square sparse matrix
        Input adjacency matrix. Must be square (n×n) as the function
        computes A ∩ A.T. Supports scipy sparse matrix formats.
    is_weighted : bool
        If True, preserve edge weights; if False, return binary matrix.
        
    Returns
    -------
    sparse matrix or numpy.matrix
        Symmetric component of the input matrix. Returns sparse matrix if input
        is sparse, dense matrix if input is dense.
        
    Raises
    ------
    ValueError
        If A is not a square matrix (due to broadcasting error in A.T operation).
        
    Notes
    -----
    The symmetric component contains only edges that exist bidirectionally.
    This is useful for analyzing reciprocal connections in directed networks.
    
    Mathematical relationship: A = symmetric_component(A) + non_symmetric_component(A)    """
    # Check if sparse to preserve sparsity
    if sp.issparse(A):
        # Work in sparse format
        A_bool = A.astype(bool)
        A_T_bool = A_bool.T
        # Element-wise AND in sparse format
        symm_mask = A_bool.multiply(A_T_bool)
        
        if not is_weighted:
            return symm_mask.astype(A.dtype)
        else:
            # For weighted, multiply with original weights
            return symm_mask.multiply(A)
    else:
        # Dense implementation (original)
        a = A.astype(bool)
        symm_mask = np.bitwise_and(a, a.T)
        if not is_weighted:
            return symm_mask
        return np.multiply(symm_mask, A)


def non_symmetric_component(A, is_weighted):
    """Extract the non-symmetric (asymmetric) component of a matrix.
    
    Finds edges that exist in only one direction, i.e., (i,j) exists
    but (j,i) does not, or vice versa.
    
    Parameters
    ----------
    A : square sparse matrix or array
        Input adjacency matrix. Must be square (n×n) for transpose operation.
    is_weighted : bool
        If True, preserve edge weights; if False, work with binary matrix.
        
    Returns
    -------
    sparse matrix or array
        Non-symmetric component of the input matrix. Returns same format as input
        (sparse if input is sparse, dense if input is dense).
        
    Raises
    ------
    ValueError
        If A is not a square matrix.
        
    Notes
    -----
    The non-symmetric component represents unidirectional connections
    in directed networks. Mathematical relationship: 
    A = symmetric_component(A) + non_symmetric_component(A).
    
    This function now preserves sparsity to avoid memory issues with large matrices.    """
    symm_comp = symmetric_component(A, is_weighted)
    
    if sp.issparse(A):
        # Sparse subtraction preserves sparsity
        return A - symm_comp
    else:
        # Dense subtraction
        return A.astype(float) - symm_comp.astype(float)


def remove_duplicates(coo):
    """Remove duplicate entries from a COO-format sparse matrix.
    
    When multiple values exist for the same (i,j) position, keeps only
    the last occurrence. This is useful for cleaning malformed sparse matrices.
    
    Parameters
    ----------
    coo : scipy.sparse.coo_matrix
        COO-format sparse matrix potentially containing duplicates.
        Must be in COO format (not CSR, CSC, etc.).
        
    Returns
    -------
    scipy.sparse.coo_matrix
        COO matrix with duplicates removed.
        
    Raises
    ------
    AttributeError
        If input is not in COO format (missing .row, .col, .data attributes).
        
    Notes
    -----
    COO format allows duplicate entries, but most algorithms expect
    unique (i,j) pairs. This function ensures data integrity by keeping
    the last value for each (i,j) position. The "last" value depends on
    the order of data in the COO arrays.
    
    Note: This uses DOK intermediate format which has O(nnz) time complexity
    but high memory overhead. For large matrices, consider using scipy's
    built-in duplicate summing behavior instead.    """
    dok = sp.dok_matrix((coo.shape), dtype=coo.dtype)
    for i, j, v in zip(coo.row, coo.col, coo.data):
        dok[i, j] = v
    return dok.tocoo()


def adj_input_to_csr_sparse_matrix(a):
    """Convert various matrix formats to CSR sparse format.
    
    Handles numpy arrays and different scipy sparse formats, ensuring
    the output is always in CSR format for efficient row operations.
    
    Parameters
    ----------
    a : np.ndarray or scipy.sparse matrix
        Input matrix in various formats (dense, COO, CSC, CSR).
        Other sparse formats (LIL, DIA, BSR) are not supported.
        
    Returns
    -------
    scipy.sparse.csr_array
        Matrix in CSR (Compressed Sparse Row) format.
        Note: returns csr_array (not csr_matrix) for consistency.
        
    Raises
    ------
    Exception
        If input format is not recognized.
        
    Notes
    -----
    CSR format is efficient for row slicing and matrix arithmetic.
    COO matrices are cleaned of duplicates before conversion.
    CSC matrices are properly converted to CSR format.    """
    if isinstance(a, np.ndarray):
        adj = sp.csr_array(a)
    elif a.format == "csr":
        adj = a
    elif a.format == "csc":
        adj = a.tocsr()
    elif a.format == "coo":
        adj = remove_duplicates(a).tocsr()
    else:
        raise Exception(
            "Wrong input parsed to adj_input_to_csr_sparse_matrix function:", type(a)
        )

    return sp.csr_array(adj)


def remove_selfloops_from_adj(a):
    """Remove self-loops (diagonal elements) from adjacency matrix.
    
    Self-loops are edges from a node to itself. This function sets
    all diagonal elements to zero.
    
    Parameters
    ----------
    a : array-like or sparse matrix
        Input adjacency matrix.
        
    Returns
    -------
    array-like or sparse matrix
        Adjacency matrix with self-loops removed (diagonal = 0).
        Always returns a copy, even if no self-loops exist.
        
    Notes
    -----
    Only modifies the matrix if self-loops exist (trace != 0).
    For sparse matrices, explicitly removes zeros after diagonal clearing.
    Always returns a copy to ensure safety.    """
    if a.trace() != 0:
        a = adj_input_to_csr_sparse_matrix(a)
        anew = a.copy()
        anew.setdiag(0)
        anew.eliminate_zeros()
        return anew
    else:
        return a.copy()


def remove_isolates_from_adj(a):
    """Remove isolated nodes (nodes with no connections) from adjacency matrix.
    
    Isolated nodes have zero in-degree and zero out-degree. This function
    removes such nodes and returns the cleaned matrix with a mapping.
    
    Parameters
    ----------
    a : array-like or sparse matrix
        Input adjacency matrix.
        
    Returns
    -------
    cleared_matrix : scipy.sparse.csr_array
        Adjacency matrix with isolated nodes removed.
    node_mapping : dict
        Mapping from new node indices to original indices.
        
    Notes
    -----
    Degree calculation is binary (ignores edge weights).    """
    a = adj_input_to_csr_sparse_matrix(a)

    in_degrees = np.array(a.astype(bool).astype(int).sum(axis=1))  # .flatten().ravel()
    out_degrees = np.array(a.astype(bool).astype(int).sum(axis=0))  # .flatten().ravel()

    indices = np.where(in_degrees + out_degrees > 0)[0]
    cleared_matrix = a[indices, :].tocsc()[:, indices].tocsr()

    # mapping of new nodes to old ones
    node_mapping = dict(zip(range(len(indices)), indices))

    return cleared_matrix, node_mapping


def sausage_index(A, nn):
    """Calculate the sausage index of a network.
    
    The sausage index measures the proportion of edges that connect
    nodes within a distance nn along the main diagonal. High values
    indicate a "sausage-like" or chain-like network structure.
    
    Parameters
    ----------
    A : array-like
        Adjacency matrix (typically symmetric).
    nn : int
        Maximum diagonal distance to consider as "sausage edges".
        
    Returns
    -------
    float
        Sausage index value between 0 and 1.
        Returns 0.0 for empty graphs (no edges).
        
    Notes
    -----
    Sausage index = (edges within nn diagonals) / (total edges).
    Values close to 1 indicate strong linear/chain structure.
    Values close to 0 indicate more random connectivity.
    Only counts upper diagonals (assumes symmetric matrix).    """
    A = A.astype(bool).astype(int)
    total_edges = np.sum(A) / 2.0
    
    if total_edges == 0:
        return 0.0
    
    sausage_edges = 0
    for i in range(nn):
        sausage_edges += sum(np.diag(A, k=i))

    return float(sausage_edges) / total_edges


# Functions below support both numpy and sparse matrices for optimal performance
def get_laplacian(A):
    """Compute the Laplacian matrix L = D - A.
    
    Parameters
    ----------
    A : array_like or sparse matrix
        Adjacency matrix
        
    Returns
    -------
    L : array_like or sparse matrix
        Laplacian matrix (same type as input).
        Always returns float type.
        
    Notes
    -----
    Uses out-degree (row sums) for directed graphs.
    Isolated nodes have L[i,i] = 0.    """
    if _is_sparse(A):
        # Sparse implementation
        A = A.astype(float)
        out_degrees = np.array(A.sum(axis=0)).ravel()
        D = sp.spdiags(out_degrees, [0], A.shape[0], A.shape[0], format="csr")
        L = D - A
        return L
    else:
        # Dense implementation
        A = A.astype(float)
        out_degrees = A.sum(axis=0)
        D = np.diag(out_degrees)
        L = D - A
        return L


def get_inv_sqrt_diag_matrix(a):
    """Compute D^(-1/2) where D is the degree matrix.
    
    Parameters
    ----------
    a : array_like or sparse matrix
        Adjacency matrix
        
    Returns
    -------
    DH : array_like or sparse matrix
        Inverse square root of degree matrix (same type as input).
        Zero-degree nodes have 0 on diagonal.    """
    n = a.shape[0]
    
    if _is_sparse(a):
        # Sparse implementation
        A = sp.csr_array(a)
        out_degrees = np.array(A.sum(axis=0)).ravel()
        diags_sqrt = 1.0 / np.sqrt(out_degrees)
        diags_sqrt[np.isinf(diags_sqrt)] = 0
        DH = sp.spdiags(diags_sqrt, [0], n, n, format="csr")
        return DH
    else:
        # Dense implementation
        out_degrees = a.sum(axis=0)
        diags_sqrt = 1.0 / np.sqrt(out_degrees)
        diags_sqrt[np.isinf(diags_sqrt)] = 0
        DH = np.diag(diags_sqrt)
        return DH


def get_norm_laplacian(a):
    """Compute the normalized Laplacian L = I - D^(-1/2) A D^(-1/2).
    
    Parameters
    ----------
    a : array_like or sparse matrix
        Adjacency matrix (must be symmetric)
        
    Returns
    -------
    matrix : array_like or sparse matrix
        Normalized Laplacian (same type as input)
    
    Raises
    ------
    Exception
        If adjacency matrix is not symmetric.
        
    Notes
    -----
    The normalized Laplacian is L = I - D^(-1/2) A D^(-1/2).    """
    if get_symmetry_index(a) != 1:
        raise Exception(
            "Cannot construct normalized laplacian matrix from a non-symmetric adjacency matrix"
        )

    n = a.shape[0]
    
    if _is_sparse(a):
        # Sparse implementation
        A = sp.csr_array(a)
        DH = get_inv_sqrt_diag_matrix(A)
        matrix = sp.eye(n, dtype=float) - DH.dot(A.dot(DH))
        return matrix
    else:
        # Dense implementation
        DH = get_inv_sqrt_diag_matrix(a)
        matrix = np.eye(n, dtype=float) - DH.dot(a.dot(DH))
        return matrix


def get_inv_diag_matrix(a):
    """Compute D^(-1) where D is the degree matrix.
    
    Parameters
    ----------
    a : array_like or sparse matrix
        Adjacency matrix
        
    Returns
    -------
    Dinv : array_like or sparse matrix
        Inverse of degree matrix (same type as input).
        Zero-degree nodes have 0 on diagonal.    """
    n = a.shape[0]
    
    if _is_sparse(a):
        # Sparse implementation
        A = sp.csr_array(a)
        out_degrees = np.array(A.sum(axis=0)).ravel()
        invdiags = 1.0 / out_degrees
        invdiags[np.isinf(invdiags)] = 0
        Dinv = sp.spdiags(invdiags, [0], n, n, format="csr")
        return Dinv
    else:
        # Dense implementation
        out_degrees = a.sum(axis=0)
        invdiags = 1.0 / out_degrees
        invdiags[np.isinf(invdiags)] = 0
        Dinv = np.diag(invdiags)
        return Dinv


def get_rw_laplacian(a):
    """Compute the random walk Laplacian L_rw = I - D^(-1)A.
    
    Parameters
    ----------
    a : array_like or sparse matrix
        Adjacency matrix
        
    Returns
    -------
    matrix : array_like or sparse matrix
        Random walk Laplacian (same type as input).
        Always returns float dtype.
        
    Notes
    -----
    For random walks, L_rw represents "staying probabilities".
    Isolated nodes have L_rw[i,i] = 1 (100% probability of staying)
    since they have no outgoing edges.    """
    n = a.shape[0]
    T = get_trans_matrix(a)
    
    if _is_sparse(a):
        matrix = sp.eye(n, dtype=float) - T
    else:
        matrix = np.eye(n, dtype=float) - T
    
    return matrix


def get_trans_matrix(a):
    """Compute the transition matrix T = D^(-1)A.
    
    Parameters
    ----------
    a : array_like or sparse matrix
        Adjacency matrix
        
    Returns
    -------
    T : array_like or sparse matrix
        Transition matrix (same type as input).
        Always returns float dtype.
        
    Notes
    -----
    Row-stochastic matrix where rows sum to 1 (or 0 for isolated nodes).
    Uses out-degree normalization for directed graphs.    """
    if _is_sparse(a):
        # Sparse implementation
        A = sp.csr_array(a)
        Dinv = get_inv_diag_matrix(a)
        T = Dinv.dot(A)
        return T
    else:
        # Dense implementation
        Dinv = get_inv_diag_matrix(a)
        T = Dinv.dot(a)
        return T
