from ._net_utils import *
from ._matrix_utils import *
from .randomization import *
from ..utils.plot import *

import scipy
from scipy import linalg as la
from scipy.sparse.linalg import eigs
from sklearn.neighbors import NearestNeighbors
from itertools import combinations
from scipy.stats import entropy
from matplotlib import cm


def check_adjacency(a):
    if a.shape[0] != a.shape[1]:
        raise Exception('Non-square adjacency matrix!')


def check_directed(directed, real_world):
    if real_world:
        if int(directed) not in [0, 1]:
            raise Exception('Fractional direction is not valid for real networks')
    elif directed < 0 or directed > 1:
        raise Exception('Wrong directed parameter value:', directed)


def check_weights_and_directions(a, weighted, directed):
    # is_directed = 1 - np.isclose((a-a.T).sum, 0)
    # is_weighted = 1 - np.isclose((a-a.astype(bool).astype(int).sum), 0)
    is_directed = not ((a != a.T).nnz == 0)
    is_weighted = not ((a != a.astype(bool).astype(int)).nnz == 0)

    if is_directed != bool(directed):
        raise Exception('Error in network construction, check directions')
    if is_weighted != weighted:
        raise Exception('Error in network construction, check weights')


class Network():
    ''''''

    def __init__(self, a, pos, network_args, spectrum_args, real_world=0,
                 coords=None, comments=0, check_connectivity=1, create_nx_graph=1):

        self.directed = network_args['directed']
        if self.directed is None:
            self.directed = not ((a != a.T).nnz == 0)

        self.weighted = network_args['weighted']
        check_directed(self.directed, real_world)

        self.real_world = real_world
        self.comments = comments
        self.network_params = network_args
        self.spectrum_params = spectrum_args
        self.create_nx_graph = create_nx_graph

        self.adj = None
        self.lap = None
        self.lap_out = None
        self.lap_in = None
        self.nlap = None

        self.adj_spectrum = None
        self.adj_eigenvectors = None
        self.lap_spectrum = None
        self.lap_eigenvectors = None
        self.lap_out_spectrum = None
        self.lap_out_eigenvectors = None
        self.lap_in_spectrum = None
        self.lap_in_eigenvectors = None
        self.nlap_spectrum = None
        self.nlap_eigenvectors = None

        self.preprocess_adj_and_pos(a, pos,
                                    check_connectivity)  # each network object has an adjacency matrix from its birth
        self.get_node_degrees()  # each network object has out- and in-degree sequences from its birth

        self.connectivity_checked = check_connectivity

        self.lem_emb = None
        self.zvalues = None
        self.ipr = None
        self.erdos_entropy = None

    def preprocess_adj_and_pos(self, a, pos, check_connectivity):
        '''
        This method is closely tied with Network.__init__() and is needed for graph preprocessing.
        Each network is characterized by an input adjacency matrix. This matrix should
        have been already created respecting requirements in self.network_params.
        In other words, it is directed/undirected and weighted/unweighted depending on initial
        settings.
        '''
        if not sp.issparse(a):
            raise Exception('non-sparse matrix parsed to network constructor!')
        # three main network characteristics define 8 types of networks
        directed = self.directed
        weighted = self.weighted
        real_world = self.real_world

        if directed:
            gtype = nx.DiGraph
        else:
            gtype = nx.Graph

        if check_connectivity:
            consensus = 0
            while not consensus:
                # check that adjacency matrix fits our requirements
                check_weights_and_directions(a, weighted, directed)

                if isinstance(a, np.ndarray):
                    res = remove_isolates_and_selfloops_from_adj(sp.csr_matrix(a), weighted, directed)
                elif isinstance(a, sp.csr_matrix):
                    res = remove_isolates_and_selfloops_from_adj(a, weighted, directed)
                elif isinstance(a, sp.coo_matrix):
                    res = remove_isolates_and_selfloops_from_adj(remove_duplicates(a), weighted, directed)
                else:
                    raise Exception('Wrong input parsed to preprocess_adj_matrix function:', type(a))

                init_g = nx.from_scipy_sparse_matrix(res, create_using=gtype)
                G = remove_isolates_and_selfloops_from_graph(init_g)  # overkill for safety
                lost_nodes = init_g.number_of_nodes() - G.number_of_nodes()
                lost_edges = init_g.number_of_edges() - G.number_of_edges()
                if lost_nodes + lost_edges != 0:
                    print('%d isolated nodes and %d selfloops killed' % (lost_nodes, lost_edges))

                #############################
                # At this point isolated nodes and selfloops have been removed from graph.
                # An isolated node is a node with zero out-, in- or both degrees, depending on algorithm.
                # Graph (or DiGraph) G contains all information from sparse matrix res and vice versa.
                #############################

                if self.comments:
                    print('Obtaining giant component...')
                gc = take_giant_component(G)  # cleared version of a graph
                gc_adj = create_adj_from_graphml(None, graph=gc, gc_checked=1, info=self.real_world,
                                                 directed=directed, weighted=weighted)

                lost_nodes = G.number_of_nodes() - gc.number_of_nodes()
                if lost_nodes > 0 and self.comments:
                    print('WARNING: %d nodes lost after giant component creation!' % lost_nodes)

                if self.real_world and self.comments:
                    print("Number of nodes in GC:", nx.number_of_nodes(gc))
                    print("Number of edges in GC: ", nx.number_of_edges(gc))
                    print('Density of GC: ', 100 * nx.density(gc), "%")

                #############################
                # Finally, we have the GC of the graph with or without zero out (in) degree nodes,
                # depending on the main matrix construction algorithm.
                # Graph (or DiGraph) gc contains all information from sparse matrix gc_adj and vice versa.
                #############################

                if a.shape == gc_adj.shape:
                    if (a != gc_adj).nnz == 0:
                        consensus = 1
                    else:
                        a = gc_adj
                else:
                    a = gc_adj
                # we need to repeat the procedure until it converges

        else:
            gc_adj = a

            if self.create_nx_graph:
                gc = nx.from_scipy_sparse_matrix(a, create_using=gtype)
                init_g = gc
            else:
                raise Exception('not implemented yet')
                gc = None

        # add node positions if provided
        if not pos is None:
            final_position_list = {node: pos[node] for node in init_g.nodes() if node in gc.nodes()}
            self.pos = final_position_list
        else:
            self.pos = None

        # self.graph = nx.from_scipy_sparse_matrix(final, create_using=nx.DiGraph())
        self.graph = gc
        self.adj = gc_adj
        self.n = nx.number_of_nodes(self.graph)

        if self.comments:
            print('Symmetry index:', get_symmetry_index(gc_adj))

    def randomize(self, rmode='adj'):

        if rmode == 'graph':
            if self.directed:
                g = nx.DiGraph(self.graph)
            else:
                g = nx.Graph(self.graph)

            new_graph = random_rewiring_IOM_preserving(g, r=10)
            a = nx.adjacency_matrix(new_graph)
            new_net = Network(a, None, self.network_params, self.spectrum_params,
                              real_world=0, comments=0)
            return new_net

        elif rmode == 'adj':
            rand_adj = adj_random_rewiring_iom_preserving(self.adj,
                                                          is_weighted=self.weighted,
                                                          r=10)

            rand_net = Network(rand_adj, None, self.network_params, self.spectrum_params,
                               real_world=0, comments=0)
            return rand_net

        elif rmode == 'complete':
            rand_adj = random_rewiring_complete_graph(self.adj)
            rand_net = Network(rand_adj, None, self.network_params, self.spectrum_params,
                               real_world=0, comments=0)
            return rand_net

        else:
            raise ValueError('Unknown randomization method')

    def get_node_degrees(self):
        # convert sparse matrix to 0-1 format and sum over specific axis
        self.outdegrees = np.array(self.adj.astype(bool).astype(int).sum(axis=0)).ravel()
        self.indegrees = np.array(self.adj.astype(bool).astype(int).sum(axis=1)).ravel()
        self.degrees = np.array((self.adj + self.adj.T).astype(bool).astype(int).sum(axis=1)).ravel()

        min_out = min(self.outdegrees)
        min_in = min(self.indegrees)
        max_out = max(self.outdegrees)
        max_in = max(self.indegrees)

        if max_out != min_out:
            self.scaled_outdegrees = np.array([1.0 * (deg - min_out) / (max_out - min_out) for deg in self.outdegrees])
        else:
            self.scaled_outdegrees = np.ones(len(self.outdegrees))

        if min_in != max_in:
            self.scaled_indegrees = np.array([1.0 * (deg - min_in) / (max_in - min_in) for deg in self.indegrees])
        else:
            self.scaled_outdegrees = np.ones(len(self.indegrees))

    def get_degree_distr(self, mode='all'):
        if mode == 'all':
            degrees = self.degrees
        elif mode == 'out':
            degrees = self.outdegrees
        elif mode == 'in':
            degrees = self.indegrees

        hist, bins = np.histogram(degrees, bins=max(degrees) - min(degrees), density=1)
        return hist

    def draw_degree_distr(self, dmode=None, cumulative=0, survival=1, log_log=0):
        if not self.directed:
            mode = 'all'

        fig, ax = create_default_figure(10,8)
        ax.set_title('Degree distribution', color='white')

        if not mode is None:
            distr = self.get_degree_distr(mode=mode)
            if cumulative:
                if survival:
                    distr = 1 - np.cumsum(distr)
                else:
                    distr = np.cumsum(distr)

            if log_log:
                degree, = ax.loglog(distr[:-1], linewidth=2, c='k', label='degree')
            else:
                degree, = ax.plot(distr, linewidth=2, c='k', label='degree')

            ax.legend(handles=[degree], fontsize=16)

        else:
            distr = self.get_degree_distr(mode='all')
            outdistr = self.get_degree_distr(mode='out')
            indistr = self.get_degree_distr(mode='in')
            distrlist = [distr, outdistr, indistr]
            if cumulative:
                if survival:
                    distrlist = [1 - np.cumsum(d) for d in distrlist]
                else:
                    distrlist = [np.cumsum(d) for d in distrlist]

            if log_log:
                degree, = ax.loglog(distrlist[0][:-1], linewidth=2, c='k', label='degree')
                outdegree, = ax.loglog(distrlist[1][:-1], linewidth=2, c='b', label='outdegree')
                indegree, = ax.loglog(distrlist[2][:-1], linewidth=2, c='r', label='indegree')
            else:
                degree, = ax.plot(distrlist[0], linewidth=2, c='k', label='degree')
                outdegree, = ax.plot(distrlist[1], linewidth=2, c='b', label='outdegree')
                indegree, = ax.plot(distrlist[2], linewidth=2, c='r', label='indegree')

            ax.legend(handles=[degree, outdegree, indegree], fontsize=16)

    def draw_spectrum(self, mode='adj', ax=None, colors=None):
        spectrum = getattr(self, mode + '_spectrum')
        if spectrum is None:
            self.diagonalize(mode=mode)

        spectrum = getattr(self, mode + '_spectrum')
        data = np.array(sorted(list(set(spectrum)), key=np.abs))

        if ax is None:
            fig, ax = create_default_figure(12,10)

        if self.directed:
            ax.scatter(data.real, data.imag, cmap='Spectral', c=colors)
        else:
            ax.hist(data.real, bins=50)

    def diagonalize(self, noise=0, mode='lap_out'):
        if self.comments:
            print('Preparing for diagonalization...')

        noise = self.spectrum_params['noise']
        neigs = self.spectrum_params['neigs']

        outdegrees = self.outdegrees
        indegrees = self.indegrees
        degrees = self.degrees

        A = self.adj.astype(float)
        n = self.n

        if n != np.count_nonzero(outdegrees) and self.connectivity_checked:
            print(n - np.count_nonzero(outdegrees), 'nodes without out-edges')
        if n != np.count_nonzero(indegrees) and self.connectivity_checked:
            print(n - np.count_nonzero(indegrees), 'nodes without in-edges')

        nz = np.count_nonzero(degrees)
        if nz != n and self.connectivity_checked:
            print('Graph has', str(n - nz), 'isolated nodes!')
            raise Exception('Graph is not connected!')

        if not self.weighted and not np.allclose(outdegrees, indegrees):
            raise Exception('out- and in- degrees do not coincide in boolean')

        if mode == 'lap' or mode == 'lap_out':
            matrix = get_laplacian(A)
        elif mode == 'nlap':
            matrix = get_norm_laplacian(A)
        elif mode == 'adj':
            matrix = A.copy()
        else:
            raise Exception(f'diagonalization not implemented for type {mode}')

        if noise == 0:
            R = np.zeros((n, n))
        else:
            R = np.multiply(matrix.A, np.random.normal(loc=0.0, scale=noise, size=(n, n)))

        if self.comments:
            print('Performing diagonalization...')

        # raw_eigs, right_eigvecs = np.linalg.eig(matrix.A + R)
        raw_eigs, right_eigvecs = la.eig(matrix + R, right=True)
        # raw_eigs, right_eigvecs = sp.linalg.eigs(matrix, which = 'LM', k=n_eigs)

        raw_eigs = np.around(raw_eigs, decimals=12)
        # print('raw eigs:', raw_eigs)
        eigs = np.sort(raw_eigs)
        if 'lap' in mode:
            n_comp = len(raw_eigs[np.abs(raw_eigs) == 0])
            if n_comp != 1 and not self.weighted and self.connectivity_checked:
                print('eigenvalues:', eigs)
                raise Exception('Graph has %d components!' % n_comp)

        setattr(self, mode + '_spectrum', eigs)
        setattr(self, mode, matrix)
        # self.eigenvectors = right_eigvecs[:,1:][np.ix_(range(len(eigs)+1), np.argsort(eigs))]
        sorted_eigenvectors = right_eigvecs[np.ix_(range(len(eigs)), np.argsort(raw_eigs))]
        setattr(self, mode + '_eigenvectors', sorted_eigenvectors)

        if self.comments:
            print('Diagonalization finished')

    def calculate_z_values(self, mode='lap_out'):
        spectrum = getattr(self, mode + '_spectrum')
        if spectrum is None:
            self.diagonalize(mode=mode)
        spectrum = getattr(self, mode + '_spectrum')

        eigs = sorted(list(set(spectrum)), key=np.abs)
        if len(eigs) != len(spectrum) and self.comments:
            print('WARNING:', len(spectrum) - len(eigs), 'repeated eigenvalues discarded')

        if self.comments:
            print('Computing nearest neighbours...')

        X = np.array([[np.real(x), np.imag(x)] for x in eigs])
        nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)
        nnbs = [eigs[x] for x in indices[:, 1]]
        nnnbs = [eigs[x] for x in indices[:, 2]]
        nndist = np.array([(nnbs[i] - eigs[i]) for i in range(len(eigs))])
        nnndist = np.array([(nnnbs[i] - eigs[i]) for i in range(len(eigs))])
        zlist = np.array([(nnbs[i] - eigs[i]) / (nnnbs[i] - eigs[i]) for i in range(len(eigs))])

        # plt.figure(figsize = (15,12))
        # plt.scatter(zlist.real,zlist.imag)
        self.zvalues = zlist

    def show(self, dtype=None, mode='adj'):
        fig, ax = plt.subplots(figsize=(16, 16))

        mat = getattr(self, mode)
        if mat is None:
            self.diagonalize(mode=mode)
        mat = getattr(self, mode)

        if not dtype is None:
            # print(self.adj.astype(dtype).A)
            ax.matshow(mat.astype(dtype).A)
        else:
            ax.matshow(mat.A)

    def draw_eigenvectors(self, left_ind, right_ind, mode='adj'):
        import matplotlib as mpl

        spectrum = getattr(self, mode + '_spectrum')
        if spectrum is None:
            self.diagonalize(mode=mode)

        spectrum = getattr(self, mode + '_spectrum')
        eigenvectors = getattr(self, mode + '_eigenvectors')

        vecs = np.real(eigenvectors[:, left_ind: right_ind + 1])
        # vecs = np.abs(self.eigenvectors[:, left_ind: right_ind+1])
        eigvals = np.real(spectrum[left_ind: right_ind + 1])

        npics = vecs.shape[1]
        pics_in_a_row = np.ceil(np.sqrt(npics))
        pics_in_a_col = np.ceil(1.0 * npics / pics_in_a_row)
        fig, ax = create_default_figure(16,12)
        # ax = fig.add_subplot(111)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,
                            hspace=0.2, wspace=0.1)

        if self.pos is None:
            # pos = nx.layout.spring_layout(self.graph)
            pos = nx.drawing.layout.circular_layout(self.graph)
        else:
            pos = self.pos

        xpos = [p[0] for p in pos.values()]
        ypos = [p[1] for p in pos.values()]

        nodesize = np.sqrt(self.scaled_outdegrees) * 100 + 10
        anchor_for_colorbar = None
        for i in range(npics):
            vec = vecs[:, i]
            ax = fig.add_subplot(pics_in_a_col, pics_in_a_row, i + 1)
            '''
            ax.set(xlim=(min(xpos)-0.1*abs(min(xpos)), max(xpos)+0.1*abs(max(xpos))),
                   ylim=(min(ypos)-0.1*abs(min(ypos)), max(ypos)+0.1*abs(max(ypos))))
            '''
            text = 'eigenvector ' + str(i + 1) + ' lambda ' + str(np.round(eigvals[i], 3))
            ax.set_title(text)
            options = {
                'node_color': vec,
                'node_size': nodesize,
                'cmap': cm.get_cmap('Spectral')
            }

            nodes = nx.draw_networkx_nodes(self.graph, pos, **options)
            if anchor_for_colorbar is None:
                anchor_for_colorbar = nodes
            # edges = nx.draw_networkx_edges(self.graph, pos, **options)
            # pc, = mpl.collections.PatchCollection(nodes, cmap = options['cmap'])
            # pc.set_array(edge_colors)
            nodes.set_clim(vmin=min(vec) * 1.1, vmax=max(vec) * 1.1)
            plt.colorbar(anchor_for_colorbar)

        plt.show()

    def draw_net(self, colors=None):
        fig, ax = plt.subplots(figsize=(16, 12))

        if self.pos is None:
            print('Node positions not found, auto layout was constructed')
            pos = nx.layout.spring_layout(self.graph)
            # pos = nx.drawing.layout.circular_layout(self.graph)
        else:
            pos = self.pos

        xpos = [p[0] for p in pos.values()]
        ypos = [p[1] for p in pos.values()]

        nodesize = np.sqrt(self.scaled_outdegrees) * 100 + 10
        options = {
            'node_size': nodesize,
            'cmap': cm.get_cmap('Spectral')
        }

        nodes = nx.draw_networkx_nodes(self.graph, pos, node_color=colors, **options)
        edges = nx.draw_networkx_edges(self.graph, pos, **options)

        plt.show()

    def calculate_ipr(self, mode='adj'):

        eigenvectors = getattr(self, mode + '_eigenvectors')
        if eigenvectors is None:
            self.diagonalize(mode=mode)
        eigenvectors = getattr(self, mode + '_eigenvectors')

        nvecs = eigenvectors.shape[1]
        ipr = np.zeros(nvecs)
        eig_entropy = np.zeros(nvecs)
        for i in range(nvecs):
            ipr[i] = sum([np.abs(v) ** 4 for v in eigenvectors[:, i]])
            # entropy[i] = -np.log(ipr[i]) # erdos entropy (deprecated)
            eig_entropy[i] = entropy(np.array([np.abs(v) ** 2 for v in eigenvectors[:, i]]))

        self.ipr = ipr
        self.eigenvector_entropy = eig_entropy / np.log(self.n)

    def calculate_von_neuman_entropy(self, corrmat_mode=0):
        if self.directed:
            raise Exception('not implemented')

        if self.lap_spectrum is None:
            self.diagonalize(mode='lap')

        degrees = self.degrees
        dg = sum(degrees)
        eigs = sorted(np.real(self.lap_spectrum))[1:]
        self.von_neuman_entropy = np.round(sum([-(l / dg) * np.log2(l / dg) for l in eigs]), 5)
        # print('von Neuman entropy:', self.von_neuman_entropy)

    def calculate_tau_entropy(self, t, verbose=0):
        if not self.directed:
            if self.lap_spectrum is None:
                self.diagonalize(mode='lap')
            eigenvalues = np.exp(-t * self.lap_spectrum)

        else:
            if self.lap_out_spectrum is None:
                self.diagonalize(mode='lap_out')
            eigenvalues = np.exp(-t * self.lap_out_spectrum)

        norm_eigenvalues = np.trim_zeros(np.real(eigenvalues / np.sum(eigenvalues)))

        if verbose:
            print('eigenvalues:', eigenvalues)
            print('norm eigenvalues:', norm_eigenvalues)

        # justification: https://journals.aps.org/prx/abstract/10.1103/PhysRevX.6.041062
        S = -np.sum(np.multiply(norm_eigenvalues, np.log2(norm_eigenvalues)))
        return S

    def calculate_free_entropy(self, t, verbose=0):
        if not self.directed:
            if self.lap_spectrum is None:
                self.diagonalize(mode='lap')
            eigenvalues = np.exp(-t * self.lap_spectrum)

        else:
            if self.lap_out_spectrum is None:
                self.diagonalize(mode='lap_out')
            eigenvalues = np.exp(-t * self.lap_out_spectrum)

        if verbose:
            print('eigenvalues:', eigenvalues)

        # justification: https://www.nature.com/articles/s42005-021-00582-8
        F = np.log2(np.real(np.sum(eigenvalues)))
        return F

    def calculate_q_entropy(self, q, t):
        if not self.directed:
            if self.lap_spectrum is None:
                self.diagonalize(mode='lap')
            eigenvalues = np.exp(-t * q * self.lap_spectrum)
            Z = np.sum(np.exp(-t * self.lap_spectrum))

        else:
            if self.lap_out_spectrum is None:
                self.diagonalize(mode='lap_out')
            eigenvalues = np.exp(-t * q * self.lap_out_spectrum)
            Z = np.sum(np.exp(-t * self.lap_out_spectrum))

        if q <= 0:
            raise Exception('q must be >0')

        else:
            if q != 1:
                S = 1 / (1 - q) * np.log(Z ** (-q) * np.sum(eigenvalues)) / np.log(2)
            else:
                norm_eigenvalues = np.trim_zeros(eigenvalues / Z)
                S = -np.real(np.sum(np.multiply(norm_eigenvalues, np.log2(norm_eigenvalues))))

        if np.imag(S) != 0:
            print(q)
            print(S)
            raise Exception('Imaginary entropy detected!')

        return np.real(S)

    def calculate_estrada_index(self):

        if self.adj_spectrum is None:
            self.diagonalize(mode='adj')

        self.estrada_index = sum([np.exp(l) for l in self.adj_spectrum])
        # print('von Neuman entropy:', self.von_neuman_entropy)

    def localization_signatures(self):
        if self.zvalues is None:
            self.calculate_z_values()

        zvals = self.zvalues
        mean_cos_phi = np.mean(np.array([np.cos(np.angle(x)) for x in zvals]))
        rvals = [1. / (np.abs(z)) ** 2 for z in zvals]
        mean_inv_r_sq = np.mean(np.array(rvals))

        if self.comments:
            print('mean cos phi complex:', mean_cos_phi)
            print('mean 1/r^2 real:', mean_inv_r_sq)

        return mean_inv_r_sq, mean_cos_phi

    def construct_lem_embedding(self, dim):
        if self.directed:
            raise Exception('LEM embedding is not implemented for directed graphs')

        A = self.adj
        A = A.asfptype()
        print('Performing spectral decomposition...')
        K = A.shape[0]  # number of nodes, each representing a data point

        diags = A.sum(axis=1).flatten()
        with scipy.errstate(divide='ignore'):
            diags_sqrt = 1.0 / scipy.sqrt(diags)
            invdiags = 1.0 / (diags)
        diags_sqrt[scipy.isinf(diags_sqrt)] = 0
        DH = scipy.sparse.spdiags(diags_sqrt, [0], K, K, format='csr')
        invD = scipy.sparse.spdiags(invdiags, [0], K, K, format='csr')

        nL = sp.eye(K) - DH.dot(A.dot(DH))
        X = A.dot(invD)

        start_v = np.ones(K)
        eigvals, eigvecs = eigs(X, k=dim + 1, which='LR', v0=start_v, maxiter=K * 1000)
        eigvals = np.asarray([np.round(np.real(x), 6) for x in eigvals])

        if np.count_nonzero(eigvals == 1.0) > 1:
            raise Exception('Error while LEM embedding construction: graph is not connected!')
        else:
            vecs = eigvecs.T[1:]
            print('max X eigs:', eigvals)
            reseigs = [np.round(1 - e, 5) for e in eigvals]

            vec_norms = np.array([np.real(sum([x * x for x in v])) for v in vecs])
            vecs = vecs / vec_norms[:, np.newaxis]
            vecs = sp.csr_matrix(vecs, dtype=float).dot(DH)
            print('min NL eigs', reseigs)

            print('Check of eigenvectors:')
            coslist = [0] * (dim)
            for i in range(dim):
                vec = vecs[i]
                cvec = vec.dot(nL)
                coslist[i] = np.round((cvec.dot(vec.T) / sp.linalg.norm(cvec) / sp.linalg.norm(vec)).sum(), 3)
                if coslist[i] != 1.0:
                    print('WARNING: cos', i + 1, ' = ', (coslist[i]))

            if sum(coslist) == int(dim):
                print('Eigenvectors confirmed')
            else:
                raise Exception('Eigenvectors not confirmed!')
                print(coslist)

            self.lem_emb = vecs

    def plot_lem_embedding(self, ndim, colors=None):

        if self.lem_emb is None:
            self.construct_lem_embedding(ndim)

        if colors is None:
            colors = np.zeros(self.lem_emb.shape[1])
            colors = range(self.lem_emb.shape[1])

        psize = 10
        data = self.lem_emb.A
        pairs = list(combinations(np.arange(ndim), 2))
        npics = len(pairs)
        pics_in_a_row = np.ceil(np.sqrt(npics))
        pics_in_a_col = np.ceil(1.0 * npics / pics_in_a_row)

        fig = plt.figure(figsize=(20, 20))
        fig.suptitle('Projections')
        for i in range(len(pairs)):
            ax = fig.add_subplot(pics_in_a_row, pics_in_a_col, i + 1)
            i1, i2 = pairs[i]
            scatter = ax.scatter(data[i1, :], data[i2, :], c=colors, s=psize)

            legend = ax.legend(*scatter.legend_elements(),
                               loc="upper left", title="Classes")

            ax.text(min(data[i1, :]), min(data[i2, :]), 'axes ' + str(i1 + 1) + ' vs.' + str(i2 + 1),
                    bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
            # ax.add_artist(legend)

def create_adj_from_graphml(datapath, graph=None, gc_checked=0, info=0,
                            directed=0, weighted=0, edges_to_delete=None,
                            nodes_to_delete=None):
    if graph is None:
        init_g = nx.read_graphml(datapath)
        # init_g = nx.convert_node_labels_to_integers(init_g)
    else:
        init_g = graph

    if not (nodes_to_delete is None):
        init_g.remove_nodes_from(nodes_to_delete)

    if not (edges_to_delete is None):
        for e in edges_to_delete:
            # print(e)
            print(init_g.has_edge(e[0], e[1]))
        init_g.remove_edges_from(edges_to_delete)

    if gc_checked:
        g = init_g
    else:
        G = remove_isolates_and_selfloops_from_graph(init_g)
        lost_nodes = init_g.number_of_nodes() - G.number_of_nodes()
        lost_edges = init_g.number_of_edges() - G.number_of_edges()
        print('%d isolated nodes and %d selfloops killed' % (lost_nodes, lost_edges))
        g = take_giant_component(G)
        lost_nodes = G.number_of_nodes() - g.number_of_nodes()
        if lost_nodes > 0:
            print('WARNING: %d nodes lost after giant component creation!' % lost_nodes)

    return nx.adjacency_matrix(g)


