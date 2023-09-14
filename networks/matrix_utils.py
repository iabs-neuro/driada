import numpy as np
import copy
import scipy.sparse as sp

def assign_random_weights(A):
    X = np.random.random(size=(A.shape[0], A.shape[0]))
    W = np.multiply(X, A)
    return (W + W.T) / 2


def turn_to_directed(mat, directed=0.0, weighted=0):
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
    # this function removes duplicate entries from a final coo matrix
    # duplicates are discarded as the data is always the same:
    # coo[i,j] = val1, coo[i,j] = val2 ==> val1 = val2

    dok = sp.dok_matrix((coo.shape), dtype=coo.dtype)
    dok._update(zip(zip(coo.row, coo.col), coo.data))
    return dok.tocoo()


def remove_isolates_and_selfloops_from_adj(a, weighted, directed, mode='lap'):
    # check for sparsity violation
    if not sp.issparse(a):
        raise Exception('Input is not sparse!')

    # remove selfloops:
    a = sp.csr_array(a)
    a.setdiag(0)
    a.eliminate_zeros()

    n_prev = a.shape[0]
    n_new = 0
    while n_new != n_prev:
        # remove nodes with zero out-, in- or both degrees:
        if weighted:
            indegrees = np.array(a.astype(bool).astype(int).sum(axis=1))[0]  # .flatten().ravel()
            outdegrees = np.array(a.astype(bool).astype(int).sum(axis=0))[0]  # .flatten().ravel()
        else:
            indegrees = np.array(a.sum(axis=1))[0]  # .flatten().ravel()
            outdegrees = np.array(a.sum(axis=0))[0]  # .flatten().ravel()

        if not directed:
            indices = np.where(indegrees + outdegrees > 0)[0]
        elif mode == 'lap_out':
            indices = np.where(outdegrees > 0)[0]
        elif mode == 'lap_in':
            indices = np.where(indegrees > 0)[0]

        cleared_matrix = a[indices, :].tocsc()[:, indices].tocsr()

        # print('shape:', cleared_matrix.shape)
        n_prev = n_new
        n_new = cleared_matrix.shape[0]
        a = cleared_matrix

    return cleared_matrix

def preprocess_adj_matrix(A, weighted, directed, info=1, mode='lap'):
    if isinstance(A, sp.csr_array):
        res = remove_isolates_and_selfloops_from_adj(A.A, weighted, directed, mode=mode)
    elif isinstance(A, sp.coo_matrix):
        res = remove_isolates_and_selfloops_from_adj(remove_duplicates(A).A, weighted, directed, mode=mode)
    else:
        print(A.dtype)
        raise Exception('Wrong input parsed to preprocess_adj_matrix function!')

    if info:
        print("Final number of nodes:", res.shape[0])
        print("Final number of edges: ", int(np.sum(res)))
        print('Density: ', np.round(200.0 * np.sum(res) / (res.shape[0]) ** 2, 3), "%")
        print('Symmetry index:', get_symmetry_index(A))
    return res


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

