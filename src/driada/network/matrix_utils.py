import numpy as np
import copy
import scipy.sparse as sp


def _plain_bfs(adj, source):
    '''
    adapted from networkx.algorithms.components.connected._plain_bfs

    Args:
        adj:
        source:

    Returns:

    '''

    n = adj.shape[0]
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
    inds = a[[node], :].nonzero()[1]
    return inds


def get_ccs_from_adj(adj):
    seen = set()
    for v in range(adj.shape[0]):
        if v not in seen:
            c = _plain_bfs(adj, v)
            seen.update(c)
            yield c


def get_sccs_from_adj(adj):
    '''
        adapted from networkx.algorithms.components.strongly_connected.strongly_connected_components
    Args:
        adj:

    Returns:

    '''

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
    connected_components = sorted(get_ccs_from_adj(adj), key=len, reverse=True)
    gcc = np.array(list(connected_components[0]))
    gcc_adj = adj[gcc, :].tocsc()[:, gcc].tocsr()

    # mapping of new nodes to old ones
    node_mapping = dict(zip(range(len(gcc)), gcc))

    return gcc_adj, node_mapping


def get_giant_scc_from_adj(adj):
    connected_components = sorted(get_sccs_from_adj(adj), key=len, reverse=True)
    gscc = np.array(list(connected_components[0]))
    gscc_adj = adj[gscc, :].tocsc()[:, gscc].tocsr()

    # mapping of new nodes to old ones
    node_mapping = dict(zip(range(len(gscc)), gscc))

    return gscc_adj, node_mapping


def assign_random_weights(A):
    X = np.random.random(size=(A.shape[0], A.shape[0]))
    W = np.multiply(X, A)
    return (W + W.T) / 2


# TODO: refactor to sparse format
def turn_to_partially_directed(mat, directed=0.0, weighted=0):
    if not isinstance(mat, np.ndarray):
        raise Exception('Wrong input parsed to turn_to_directed function!')

    A = copy.deepcopy(mat)

    if directed == 0.0 or directed is None:
        if not weighted:
            a = A.astype(bool)
        else:
            a = A.astype(float)

        return sp.csr_array(a)

    np.fill_diagonal(A, 0)
    rows, cols = A.nonzero()
    edgeset = set(zip(rows, cols))
    upper = np.array([l for l in edgeset if l[0] < l[1]])
    dircount = 0

    random_tosses = np.random.random(len(upper))
    condition1 = (random_tosses >= directed / 2.0) & (random_tosses < directed)
    condition2 = (random_tosses <= directed / 2.0) & (random_tosses < directed)
    indices_where_upper_is_removed = np.where(condition1 == True)[0]
    indices_where_lower_is_removed = np.where(condition2 == True)[0]

    u_xdata = [u[0] for u in upper[indices_where_upper_is_removed]]
    u_ydata = [u[1] for u in upper[indices_where_upper_is_removed]]
    A[u_xdata, u_ydata] = 0

    l_xdata = [u[1] for u in upper[indices_where_lower_is_removed]]
    l_ydata = [u[0] for u in upper[indices_where_lower_is_removed]]
    A[l_xdata, l_ydata] = 0

    '''
    for i in range(double_edges):
        toss = random.random()
        if toss < directed: # this means double edge will be reduced to single randomly
            dircount += 1
            if toss >= directed/2.:
                A[upper_right[i]] = 0#A[upper_right[i][::-1]] + 0#.1*np.random.random()
            else:
                A[upper_right[i][::-1]] = 0#A[upper_right[i]] + 0#.1*np.random.random()
    '''

    #a = sp.csr_array(A)
    # get_symmetry_index(a)
    return A


def get_symmetry_index(a):
    a = a.astype(bool)
    symmetrized = a + a.T

    difference = symmetrized.astype(int) - a.astype(int)
    difference.eliminate_zeros()
    symm_index = 1 - difference.nnz / symmetrized.nnz * 2
    # symm_index is 1 for a symmetrix matrix and 0 for an asymmetric one
    return symm_index


def symmetric_component(A, is_weighted):
    a = A.astype(bool).A
    symm_mask = np.bitwise_and(a, a.T)
    if not is_weighted:
        return symm_mask

    return np.multiply(symm_mask, A.A)


def non_symmetric_component(A, is_weighted):
    return A.astype(float) - symmetric_component(A, is_weighted).astype(float)


def remove_duplicates(coo):
    # this function removes duplicate entries from a coo-format adjacency matrix
    # duplicates are discarded as the data is always the same:
    # coo[i,j] = val1, coo[i,j] = val2 ==> val1 = val2

    dok = sp.dok_matrix((coo.shape), dtype=coo.dtype)
    dok._update(zip(zip(coo.row, coo.col), coo.data))
    return dok.tocoo()


def adj_input_to_csr_sparse_matrix(a):
    if isinstance(a, np.ndarray):
        adj = sp.csr_array(a)
    elif a.format in ['csr', 'csc']:
        adj = a
    elif a.format == 'coo':
        adj = remove_duplicates(a)
    else:
        raise Exception('Wrong input parsed to preprocess_adj_matrix function:', type(a))

    return sp.csr_array(adj)


def remove_selfloops_from_adj(a):
    if a.trace() != 0:
        a = adj_input_to_csr_sparse_matrix(a)
        anew = a.copy()
        anew.setdiag(0)
        anew.eliminate_zeros()
        return anew
    else:
        return a


def remove_isolates_from_adj(a):
    a = adj_input_to_csr_sparse_matrix(a)

    in_degrees = np.array(a.astype(bool).astype(int).sum(axis=1))  # .flatten().ravel()
    out_degrees = np.array(a.astype(bool).astype(int).sum(axis=0))  # .flatten().ravel()

    indices = np.where(in_degrees + out_degrees > 0)[0]
    cleared_matrix = a[indices, :].tocsc()[:, indices].tocsr()

    # mapping of new nodes to old ones
    node_mapping = dict(zip(range(len(indices)), indices))

    return cleared_matrix, node_mapping


def sausage_index(A, nn):
    A = A.astype(bool).astype(int)
    sausage_edges = 0
    for i in range(nn):
        sausage_edges += sum(np.diag(A, k=i))

    si = sausage_edges / (np.sum(A) / 2)
    print('sausage edges:', sausage_edges)
    print('other edges:', np.sum(A) / 2 - sausage_edges)
    print('sausage index=', si)


#TODO: create separate branches for np and sparse matrices for further functions
def get_laplacian(A):
    A = A.astype(float)
    out_degrees = np.array(A.sum(axis=0)).ravel()
    D = sp.spdiags(out_degrees, [0], A.shape[0], A.shape[0], format='csr')
    L = D - A
    return L


def get_inv_sqrt_diag_matrix(a):
    n = a.shape[0]
    A = sp.csr_array(a)
    out_degrees = np.array(A.sum(axis=0)).ravel()
    diags_sqrt = 1.0 / np.sqrt(out_degrees)

    diags_sqrt[np.isinf(diags_sqrt)] = 0
    DH = sp.spdiags(diags_sqrt, [0], n, n, format='csr')
    return DH


def get_norm_laplacian(a):
    if get_symmetry_index(a) != 1:
        raise Exception('Cannot construct normalized laplacian matrix from a non-hermitian adjacency matrix')

    n = a.shape[0]
    A = sp.csr_array(a)
    DH = get_inv_sqrt_diag_matrix(A)
    matrix = sp.eye(n, dtype=float) - DH.dot(A.dot(DH))
    return matrix


def get_inv_diag_matrix(a):
    n = a.shape[0]
    A = sp.csr_array(a)
    out_degrees = np.array(A.sum(axis=0)).ravel()
    invdiags = 1.0 / out_degrees

    invdiags[np.isinf(invdiags)] = 0
    Dinv = sp.spdiags(invdiags, [0], n, n, format='csr')
    return Dinv


def get_rw_laplacian(a):
    n = a.shape[0]
    T = get_trans_matrix(a)
    matrix = sp.eye(n, dtype=float) - T
    return matrix


def get_trans_matrix(a):
    A = sp.csr_array(a)
    Dinv = get_inv_diag_matrix(a)
    T = Dinv.dot(A)
    return T
