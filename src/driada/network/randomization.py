from .graph_utils import *
from .matrix_utils import *
import warnings
import random

def adj_random_rewiring_iom_preserving(a, is_weighted, r=10):
    # print ('Rewiring double connections...')

    s = symmetric_component(a, is_weighted)
    rs = turn_to_partially_directed(s, directed=1.0, weighted=is_weighted)
    rows, cols = rs.todense().nonzero()
    edgeset = set(zip(rows, cols))
    upper = [l for l in edgeset]  # if l[0]<l[1]]
    source_nodes = [e[0] for e in upper]
    target_nodes = [e[1] for e in upper]

    double_edges = len(upper)

    i = 0

    while i < double_edges * r:
        # print('i=',i)
        good_choice = 0
        while not good_choice:
            ind1, ind2 = np.random.choice(double_edges, 2)
            n1, n3 = source_nodes[ind1], source_nodes[ind2]
            n2, n4 = target_nodes[ind1], target_nodes[ind2]

            if len(set([n1, n2, n3, n4])) == 4:
                good_choice = 1

        w1 = s[n1, n2]
        w2 = s[n2, n1]
        w3 = s[n3, n4]
        w4 = s[n4, n3]

        '''
        if w1*w2*w3*w4 == 0:
            print(i, w1, w2, w3, w4)
        '''
        if s[n1, n3] + s[n1, n4] + s[n2, n3] + s[n2, n4] == 0:
            s[n1, n4] = w1
            s[n4, n1] = w2
            s[n2, n3] = w3
            s[n3, n2] = w4

            s[n1, n2] = 0
            s[n2, n1] = 0
            s[n3, n4] = 0
            s[n4, n3] = 0

            target_nodes[ind1], target_nodes[ind2] = n4, n2

            i += 1
            # print(get_symmetry_index(sp.csr_array(A)))

    # plt.matshow(s)
    # print ('Rewiring single connections...')

    ns = non_symmetric_component(a, is_weighted)

    # plt.matshow(ns)
    rows, cols = ns.nonzero()
    edges = list((set(zip(rows, cols))))
    source_nodes = [e[0] for e in edges]
    target_nodes = [e[1] for e in edges]
    single_edges = len(edges)

    i = 0

    while i < single_edges * r:
        # while i < 10:
        good_choice = 0
        while not good_choice:
            ind1, ind2 = np.random.choice(single_edges, 2)
            n1, n3 = source_nodes[ind1], source_nodes[ind2]
            n2, n4 = target_nodes[ind1], target_nodes[ind2]

            if len(set([n1, n2, n3, n4])) == 4:
                good_choice = 1

        w1 = ns[n1, n2]
        w2 = ns[n3, n4]

        checklist = [ns[n1, n3], ns[n1, n4], ns[n2, n3], ns[n2, n4],
                     ns[n3, n1], ns[n4, n1], ns[n3, n2], ns[n4, n2],
                     s[n3, n1], s[n4, n1], s[n3, n2], s[n4, n2]]

        if checklist.count(0) == 12:
            ns[n1, n4] = w1
            ns[n3, n2] = w2

            ns[n1, n2] = 0
            ns[n3, n4] = 0

            i += 1

            target_nodes[ind1], target_nodes[ind2] = n4, n2
            # print(get_symmetry_index(sp.csr_array(A)))

    res = s + ns
    if not is_weighted:
        res = res.astype(bool)

    return sp.csr_array(res)


def random_rewiring_complete_graph(a, p=1.0):
    # p ranges from 0 to 1 and defines the degree of reshuffling (percent of edges affected)
    n = a.shape[0]
    if len(np.nonzero(a)[0]) < n ** 2 - n:
        raise Exception('Graph is not complete')

    symmetric = np.allclose(a, a.T)

    all_x_inds, all_y_inds = np.nonzero(a)
    vals = a[all_x_inds, all_y_inds]
    shuffled_positions = np.random.choice(np.arange((n ** 2 - n)),
                                          replace=False,
                                          size=int(p * (n ** 2 - n)))

    shuffled_x_inds = all_x_inds[shuffled_positions]
    shuffled_y_inds = all_y_inds[shuffled_positions]
    shuffled_vals = vals[shuffled_positions]
    np.random.shuffle(shuffled_vals)  # shuffling in-place

    stable_part = a.copy()
    if symmetric:
        stable_part[shuffled_x_inds, shuffled_y_inds] = 0
        stable_part[shuffled_y_inds, shuffled_x_inds] = 0
    else:
        stable_part[shuffled_x_inds, shuffled_y_inds] = 0

    if symmetric:
        shuffled_half = np.zeros(a.shape)
        shuffled_half[shuffled_x_inds, shuffled_y_inds] = shuffled_vals
        shuffled_part = (shuffled_half + shuffled_half.T) / 2.0
    else:
        shuffled_part = np.zeros(a.shape)
        shuffled_part[shuffled_x_inds, shuffled_y_inds] = shuffled_vals

    rewired = stable_part + shuffled_part

    return rewired


def random_rewiring_dense_graph(a):
    if isinstance(a, np.ndarray):
        afull = a
        nelem = len(np.nonzero(a)[0])
    else:
        afull = a.todense()
        nelem = a.nnz

    if nelem != a.shape[0] ** 2 - a.shape[0]:
        # warnings.warn('Graph is not complete, proceeding with gap filling trick')
        temp = 0.0001
    else:
        temp = 0

    psA = afull + np.full(a.shape, temp) - np.eye(a.shape[0]) * temp
    vals = psA[np.nonzero(np.triu(psA))]
    np.random.shuffle(vals)

    tri = np.zeros(a.shape)
    tri[np.triu_indices(a.shape[0], 1)] = vals
    rewired = tri + tri.T
    res = rewired - np.full(a.shape, temp) + np.eye(a.shape[0]) * temp

    return res


def get_single_double_edges_lists(g):
    L1 = []
    L2 = []
    h = nx.to_undirected(g).copy()
    for e in h.edges():
        if g.has_edge(e[1], e[0]):
            if g.has_edge(e[0], e[1]):
                L2.append((e[0], e[1]))
            else:
                L1.append((e[1], e[0]))
        else:
            L1.append((e[0], e[1]))

    return [L1, L2]


# warning: the code below is legacy and may be inefficient
def random_rewiring_IOM_preserving(G, r=10):

    [L1, L2] = get_single_double_edges_lists(G)
    Number_of_single_edges = len(L1)
    Number_of_double_edges = len(L2)
    Number_of_rewired_1_edge_pairs = Number_of_single_edges * r
    Number_of_rewired_2_edge_pairs = Number_of_double_edges * r
    # Number_of_rewired_2_edge_pairs = 20
    i = 0
    count = 0
    previous_text = ''

    # print len(set(L).intersection(List_of_edges))
    print('Rewiring double connections...')
    while i < Number_of_rewired_2_edge_pairs:
        Edge_index_1 = random.randint(0, Number_of_double_edges - 1)
        Edge_index_2 = random.randint(0, Number_of_double_edges - 1)
        Edge_1 = L2[Edge_index_1]
        Edge_2 = L2[Edge_index_2]
        [Node_A, Node_B] = Edge_1
        [Node_C, Node_D] = Edge_2
        while (Node_A == Node_C) or (Node_A == Node_D) or (Node_B == Node_C) or (Node_B == Node_D):
            Edge_index_1 = random.randint(0, Number_of_double_edges - 1)
            Edge_index_2 = random.randint(0, Number_of_double_edges - 1)
            Edge_1 = L2[Edge_index_1]
            Edge_2 = L2[Edge_index_2]
            [Node_A, Node_B] = Edge_1
            [Node_C, Node_D] = Edge_2

        # print ('Edges:',Node_A, Node_B, ';',Node_C, Node_D)
        # print G.has_edge(Node_A, Node_B), G.has_edge(Node_B, Node_A), G.has_edge(Node_C, Node_D), G.has_edge(Node_D, Node_C)
        if G.has_edge(Node_A, Node_D) == 0 and G.has_edge(Node_D, Node_A) == 0 and G.has_edge(Node_C,
                                                                                              Node_B) == 0 and G.has_edge(
                Node_B, Node_C) == 0:
            # try:
            try:
                w_ab = G.get_edge_data(Node_A, Node_B)['weight']
            except:
                pass
            G.remove_edge(Node_A, Node_B)
            G.remove_edge(Node_B, Node_A)
            '''
            except nx.NetworkXError:
                pass
                #print('fuck')
            '''
            try:
                try:
                    w_cd = G.get_edge_data(Node_C, Node_D)['weight']
                except:
                    pass
                G.remove_edge(Node_C, Node_D)
                G.remove_edge(Node_D, Node_C)
            except nx.NetworkXError:
                pass
                # print('fuck')

            try:
                G.add_edge(Node_A, Node_D, weight=w_ab)
                G.add_edge(Node_D, Node_A, weight=w_ab)
            except:
                G.add_edge(Node_A, Node_D)
                G.add_edge(Node_D, Node_A)

            try:
                G.add_edge(Node_C, Node_B, weight=w_cd)
                G.add_edge(Node_B, Node_C, weight=w_cd)
            except:
                G.add_edge(Node_C, Node_B)
                G.add_edge(Node_B, Node_C)

            # print L2[Edge_index_1]
            L2[Edge_index_1] = (Node_A, Node_D)
            # print L2[Edge_index_1]
            # L2[Edge_index_1+1] = (Node_D, Node_A)
            L2[Edge_index_2] = (Node_C, Node_B)
            # L2[Edge_index_2+1] = (Node_B, Node_C)
            i += 1

        if (i != 0) and (i % (Number_of_double_edges // 1)) == 0:
            text = str(round(100.0 * i / Number_of_rewired_2_edge_pairs, 0)) + "%"
            if text != previous_text:
                # print text
                pass
            previous_text = text

    i = 0
    print('Rewiring single connections...')
    while i < Number_of_rewired_1_edge_pairs:
        Edge_index_1 = random.randint(0, Number_of_single_edges - 1)
        Edge_index_2 = random.randint(0, Number_of_single_edges - 1)
        Edge_1 = L1[Edge_index_1]
        Edge_2 = L1[Edge_index_2]
        [Node_A, Node_B] = Edge_1
        [Node_C, Node_D] = Edge_2
        while (Node_A == Node_C) or (Node_A == Node_D) or (Node_B == Node_C) or (Node_B == Node_D):
            Edge_index_1 = random.randint(0, Number_of_single_edges - 1)
            Edge_index_2 = random.randint(0, Number_of_single_edges - 1)
            Edge_1 = L1[Edge_index_1]
            Edge_2 = L1[Edge_index_2]
            [Node_A, Node_B] = Edge_1
            [Node_C, Node_D] = Edge_2

        if G.has_edge(Node_A, Node_D) == 0 and G.has_edge(Node_D, Node_A) == 0 and G.has_edge(Node_C,
                                                                                              Node_B) == 0 and G.has_edge(
                Node_B, Node_C) == 0:
            try:
                try:
                    w_ab = G.get_edge_data(Node_A, Node_B)['weight']
                except:
                    pass
                G.remove_edge(Node_A, Node_B)


            except nx.NetworkXError:
                print('fuck')

            try:
                try:
                    w_cd = G.get_edge_data(Node_C, Node_D)['weight']
                except:
                    pass
                G.remove_edge(Node_C, Node_D)

            except nx.NetworkXError:
                print('fuck')

            try:
                G.add_edge(Node_A, Node_D, weight=w_ab)
            except:
                G.add_edge(Node_A, Node_D)

            try:
                G.add_edge(Node_C, Node_B, weight=w_cd)
            except:
                G.add_edge(Node_C, Node_B)

            L1[Edge_index_1] = (Node_A, Node_D)
            L1[Edge_index_2] = (Node_C, Node_B)
            i += 1

        if (i != 0) and (i % (Number_of_single_edges // 1)) == 0:
            text = str(round(100.0 * i / Number_of_rewired_1_edge_pairs, 0)) + "%"
            if text != previous_text:
                # print text
                pass
            previous_text = text

    G_rewired = copy.deepcopy(G)

    return G_rewired