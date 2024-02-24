# from .matrix_utils import *
from .randomization import *
from .drawing import *
from .spectral import *
from .quantum import *

from scipy import linalg as la
from scipy.sparse.linalg import eigs
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy

UNDIR_MATRIX_TYPES = ['adj', 'trans', 'lap', 'nlap', 'rwlap']
DIR_MATRIX_TYPES = ['adj', 'lap_out', 'lap_in']
MATRIX_TYPES = UNDIR_MATRIX_TYPES + DIR_MATRIX_TYPES
SUPPORTED_GRAPH_TYPES = [nx.Graph, nx.DiGraph]


def check_matrix_type(mode, is_directed):
    if mode not in MATRIX_TYPES:
        raise ValueError(f'Matrix type {mode} is not in allowed matrix types: {MATRIX_TYPES}')

    if is_directed and mode not in DIR_MATRIX_TYPES:
        raise ValueError(f'Matrix type {mode} is not allowed for directed networks.'
                         f'Supported options are: {DIR_MATRIX_TYPES}')

    if not is_directed and mode not in UNDIR_MATRIX_TYPES:
        raise ValueError(f'Matrix type {mode} is not allowed for undirected networks.'
                         f'Supported options are: {UNDIR_MATRIX_TYPES}')


def check_adjacency(a):
    if a.shape[0] != a.shape[1]:
        raise Exception('Non-square adjacency matrix!')


def check_directed(directed, real_world):
    if real_world:
        if int(directed) not in [0, 1]:
            raise Exception('Fractional direction is not valid for a real network')
    elif directed < 0 or directed > 1:
        raise Exception('Wrong "directed" parameter value:', directed)


def check_weights_and_directions(a, weighted, directed):
    is_directed = not np.allclose(a.data, a.T.data)
    is_weighted = not np.allclose(a.data, a.astype(bool).astype(int).data)

    symm_text = 'asymmetric' if is_directed else 'symmetric'
    if is_directed != bool(directed):
        raise Exception(f'Error in network construction: "directed" set to {directed},'
                        f' but the adjacency matrix is {symm_text}')

    w_text = 'weighted' if is_weighted else 'not weighted'
    if is_weighted != bool(weighted):
        raise Exception(f'Error in network construction: "weighted" set to {weighted},'
                        f' but the adjacency matrix {w_text}')


def select_construction_pipeline(a, graph):
    # TODO: add partial directions
    if a is None:
        if graph is None:
            raise ValueError('Either "adj" or "graph" argument must be non-empty')
        else:
            if not np.any([isinstance(graph, gtype) for gtype in SUPPORTED_GRAPH_TYPES]):
                raise TypeError(f'graph should have one of supported graph types: {SUPPORTED_GRAPH_TYPES}')
            else:
                pipeline = 'graph'

    else:
        if graph is None:
            pipeline = 'adj'
        else:
            raise ValueError('Either "adj" or "graph" should be given, not both')

    return pipeline


class Network:
    """
    An object for network analysis with the focus on spectral graph theory
    """

    def __init__(self,
                 adj=None,
                 graph=None,
                 preprocessing='giant_cc',
                 name='',
                 pos=None,
                 verbose=False,
                 create_nx_graph=True,
                 node_attrs=None,
                 **network_args):

        self.name = name
        self.verbose = verbose
        self.network_params = network_args
        self.create_nx_graph = create_nx_graph

        self.init_method = select_construction_pipeline(adj, graph)

        self.directed = network_args.get('directed')
        if self.directed is None:
            if self.init_method == 'adj':
                self.directed = not np.allclose(adj.data, adj.T.data)
            elif self.init_method == 'graph':
                self.directed = nx.is_directed(graph)

        self.weighted = network_args.get('weighted')
        if self.weighted is None:
            if self.init_method == 'adj':
                self.weighted = not np.allclose(adj.data, adj.astype(bool).astype(int).data)
            elif self.init_method == 'graph':
                self.weighted = nx.is_weighted(graph)

        self.real_world = network_args.get('real_world')
        if self.real_world is None:
            self.real_world = True

        check_directed(self.directed, self.real_world)

        # set empty attributes for different matrix and data types
        valid_mtypes = DIR_MATRIX_TYPES if self.directed else UNDIR_MATRIX_TYPES
        for mt in valid_mtypes:
            setattr(self, mt, None)
            setattr(self, mt + '_spectrum', None)
            setattr(self, mt + '_eigenvectors', None)
            setattr(self, mt + '_zvalues', None)
            setattr(self, mt + '_ipr', None)

        # initialize adjacency matrix and (probably) associated graph
        if self.init_method == 'adj':
            # initialize Network object from sparse matrix
            self._preprocess_adj_and_data(a=adj,
                                          pos=pos,
                                          node_attrs=node_attrs,
                                          preprocessing=preprocessing,
                                          create_graph=create_nx_graph)

        if self.init_method == 'graph':
            # initialize Network object from NetworkX graph or digraph
            self._preprocess_graph_and_data(graph=graph,
                                            pos=pos,
                                            node_attrs=node_attrs,
                                            preprocessing=preprocessing)

        # each network object has out- and in-degree sequences from its initialization
        self.get_node_degrees()

        # TODO: remove lem_emb
        self.lem_emb = None

    def _preprocess_graph_and_data(self,
                                   graph=None,
                                   preprocessing=None,
                                   pos=None,
                                   node_attrs=None):

        if preprocessing is None:
            if self.verbose:
                print('No preprocessing specified, this may lead to unexpected errors in graph connectivity!')
            fgraph = remove_selfloops_from_graph(graph)

        elif preprocessing == 'remove_isolates':
            fgraph = remove_isolates_and_selfloops_from_graph(graph)

        elif preprocessing == 'giant_cc':
            g_ = remove_selfloops_from_graph(graph)
            fgraph = get_giant_cc_from_graph(g_)
            self.n_cc = 1

        elif preprocessing == 'giant_scc':
            g_ = remove_selfloops_from_graph(graph)
            fgraph = get_giant_scc_from_graph(g_)
            self.n_cc = 1
            self.n_scc = 1

        else:
            raise ValueError('Wrong preprocessing type!')

        lost_nodes = graph.number_of_nodes() - fgraph.number_of_nodes()
        lost_edges = graph.number_of_edges() - fgraph.number_of_edges()
        if lost_nodes + lost_edges != 0 and self.verbose:
            print(f'{lost_nodes} nodes and {lost_edges} edges removed ')

        # add node positions if provided
        if pos is not None:
            self.pos = {node: pos[node] for node in graph.nodes() if node in fgraph.nodes()}
        else:
            self.pos = None

        if node_attrs is not None:
            self.node_attrs = {node: node_attrs[node] for node in graph.nodes() if node in fgraph.nodes()}
        else:
            self.node_attrs = None

        self.graph = fgraph
        self.adj = nx.adjacency_matrix(fgraph)
        self.n = nx.number_of_nodes(self.graph)

    def _preprocess_adj_and_data(self,
                                 a=None,
                                 preprocessing=None,
                                 pos=None,
                                 node_attrs=None,
                                 create_graph=True):

        # if NetworkX graph should be created, we revert to graph-based initialization for simplicity
        if create_graph:
            gtype = nx.DiGraph if self.directed else nx.Graph
            graph = nx.from_scipy_sparse_array(a, create_using=gtype)
            self._preprocess_graph_and_data(graph=graph,
                                            pos=pos,
                                            node_attrs=node_attrs,
                                            preprocessing=preprocessing)

            return

        if preprocessing is None:
            if self.verbose:
                print('No preprocessing specified, this may lead to unexpected errors in graph connectivity!')
            fadj = remove_selfloops_from_adj(a)
            nodes_range = range(fadj.shape[0])
            node_mapping = dict(zip(nodes_range, nodes_range))  # no nodes have been deleted

        elif preprocessing == 'remove_isolates':
            a_ = remove_selfloops_from_adj(a)
            fadj, node_mapping = remove_isolates_from_adj(a_)

        elif preprocessing == 'giant_cc':
            a_ = remove_selfloops_from_adj(a)
            fadj, node_mapping = get_giant_cc_from_adj(a_)
            self.n_cc = 1

        elif preprocessing == 'giant_scc':
            a_ = remove_selfloops_from_adj(a)
            fadj, node_mapping = get_giant_scc_from_adj(a_)
            self.n_cc = 1
            self.n_scc = 1

        else:
            raise ValueError('Wrong preprocessing type!')

        lost_nodes = a.shape[0] - fadj.shape[0]
        lost_edges = a.nnz - fadj.nnz
        if not self.directed:
            lost_edges = lost_edges // 2

        if lost_nodes + lost_edges != 0 and self.verbose:
            print(f'{lost_nodes} nodes and {lost_edges} edges removed')

        # add node positions if provided
        if pos is not None:
            self.pos = {node: pos[node] for node in range(a.shape[0]) if node in node_mapping}
        else:
            self.pos = None

        if node_attrs is not None:
            self.node_attrs = {node: node_attrs[node] for node in range(a.shape[0]) if node in node_mapping}
        else:
            self.node_attrs = None

        self.graph = None
        self.adj = fadj
        self._init_to_final_node_mapping = node_mapping
        self.n = self.adj.shape[0]

    def is_connected(self):
        ccs = list(get_ccs_from_adj(self.adj))
        return len(ccs) == 1

    def randomize(self, rmode='shuffle'):
        # TODO: update routines
        if rmode == 'graph_iom':
            if self.directed:
                g = nx.DiGraph(self.graph)
            else:
                g = nx.Graph(self.graph)

            new_graph = random_rewiring_IOM_preserving(g, r=10)
            rand_adj = nx.adjacency_matrix(new_graph)

        elif rmode == 'adj_iom':
            rand_adj = adj_random_rewiring_iom_preserving(self.adj,
                                                          is_weighted=self.weighted,
                                                          r=10)

        elif rmode == 'complete':
            rand_adj = random_rewiring_complete_graph(self.adj)

        elif rmode == 'shuffle':
            rand_adj = random_rewiring_dense_graph(self.adj)

        else:
            raise ValueError('Unknown randomization method')

        rand_net = Network(sp.csr_matrix(rand_adj),
                           self.network_params,
                           name=self.name + f' {rmode} rand',
                           pos=self.pos,
                           real_world=False,
                           verbose=False)

        return rand_net

    def get_node_degrees(self):
        # convert sparse matrix to 0-1 format and sum over specific axis
        self.outdeg = np.array(self.adj.astype(bool).astype(int).sum(axis=0)).ravel()
        self.indeg = np.array(self.adj.astype(bool).astype(int).sum(axis=1)).ravel()
        self.deg = np.array((self.adj + self.adj.T).astype(bool).astype(int).sum(axis=1)).ravel()

        min_out = min(self.outdeg)
        min_in = min(self.indeg)
        max_out = max(self.outdeg)
        max_in = max(self.indeg)

        if max_out != min_out:
            self.scaled_outdeg = np.array([1.0 * (deg - min_out) / (max_out - min_out) for deg in self.outdeg])
        else:
            self.scaled_outdeg = np.ones(len(self.outdeg))

        if min_in != max_in:
            self.scaled_indeg = np.array([1.0 * (deg - min_in) / (max_in - min_in) for deg in self.indeg])
        else:
            self.scaled_outdeg = np.ones(len(self.indeg))

    def get_degree_distr(self, mode='all'):
        if mode == 'all':
            deg = self.deg
        elif mode == 'out':
            deg = self.outdeg
        elif mode == 'in':
            deg = self.indeg
        else:
            raise ValueError('Wrong mode for degree distribution.')

        hist, bins = np.histogram(deg, bins=max(deg) - min(deg), density=True)
        return hist

    def get_matrix(self, mode):
        check_matrix_type(mode, self.directed)
        matrix = getattr(self, mode)
        if matrix is None:
            if mode == 'lap' or mode == 'lap_out':
                matrix = get_laplacian(self.adj)
            elif mode == 'nlap':
                matrix = get_norm_laplacian(self.adj)
            elif mode == 'rwlap':
                matrix = get_rw_laplacian(self.adj)
            elif mode == 'trans':
                matrix = get_trans_matrix(self.adj)
            elif mode == 'adj':
                matrix = self.adj
            else:
                raise Exception(f'Wrong matrix type: {mode}')

            setattr(self, mode, matrix)

        matrix = getattr(self, mode)
        return matrix

    def get_spectrum(self, mode):
        check_matrix_type(mode, self.directed)
        spectrum = getattr(self, mode + '_spectrum')
        if spectrum is None:
            self.diagonalize(mode=mode)
            spectrum = getattr(self, mode + '_spectrum')

        return spectrum

    def get_eigenvectors(self, mode):
        check_matrix_type(mode, self.directed)
        eigenvectors = getattr(self, mode + '_eigenvectors')
        if eigenvectors is None:
            self.diagonalize(mode=mode)
            eigenvectors = getattr(self, mode + '_eigenvectors')

        return eigenvectors

    def get_ipr(self, mode):
        check_matrix_type(mode, self.directed)
        ipr = getattr(self, mode + '_ipr')
        if ipr is None:
            self.calculate_ipr(mode=mode)
            ipr = getattr(self, mode + '_ipr')

        return ipr

    def get_z_values(self, mode):
        check_matrix_type(mode, self.directed)
        zvals = getattr(self, mode + '_zvalues')
        if zvals is None:
            self.calculate_z_values(mode=mode)
            zvals = getattr(self, mode + '_zvalues')

        return zvals

    def partial_diagonalize(self, spectrum_params):
        '''
        noise = self.spectrum_params['noise']
        neigs = self.spectrum_params['neigs']
        if noise == 0:
            R = np.zeros((n, n))
        else:
            R = np.multiply(matrix.A, np.random.normal(loc=0.0, scale=noise, size=(n, n)))
        '''
        raise Exception('this method is under construction')

    def diagonalize(self, mode='lap', verbose=None):
        if verbose is None:
            verbose = self.verbose

        check_matrix_type(mode, self.directed)
        if verbose:
            print('Preparing for diagonalizing...')

        outdeg = self.outdeg
        indeg = self.indeg
        deg = self.deg

        A = self.adj.astype(float)
        n = self.n

        if n != np.count_nonzero(outdeg) and verbose:
            print(n - np.count_nonzero(outdeg), 'nodes without out-edges')
        if n != np.count_nonzero(indeg) and verbose:
            print(n - np.count_nonzero(indeg), 'nodes without in-edges')

        nz = np.count_nonzero(deg)
        if nz != n and self.n_cc == 1:
            print('Graph has', str(n - nz), 'isolated nodes!')
            raise Exception('Graph is not connected!')

        if not self.weighted and not self.directed and not np.allclose(outdeg, indeg):
            raise Exception('out- and in- degrees do not coincide in boolean')

        matrix = self.get_matrix(mode)

        if verbose:
            print('Performing diagonalization...')

        matrix_is_symmetric = np.allclose(matrix.data, matrix.T.data)
        if matrix_is_symmetric:
            raw_eigs, right_eigvecs = la.eigh(matrix.A)
        else:
            raw_eigs, right_eigvecs = la.eig(matrix.A, right=True)

        raw_eigs = np.around(raw_eigs, decimals=12)
        sorted_eigs = np.sort(raw_eigs)

        if 'lap' in mode:
            n_comp = len(raw_eigs[np.abs(raw_eigs) == 0])
            if n_comp != 1 and not self.weighted and self.n_cc == 1:
                print('eigenvalues:', sorted_eigs)
                raise Exception('Graph has %d components!' % n_comp)

        setattr(self, mode, matrix)

        if np.allclose(np.imag(sorted_eigs), np.zeros(len(sorted_eigs)), atol=1e-12):
            sorted_eigs = np.real(sorted_eigs)
        else:
            if not self.directed:
                raise ValueError('Complex eigenvalues found in non-directed network!')

        setattr(self, mode + '_spectrum', sorted_eigs)

        sorted_eigenvectors = right_eigvecs[np.ix_(range(len(sorted_eigs)), np.argsort(raw_eigs))]
        if np.allclose(np.imag(sorted_eigenvectors), np.zeros(sorted_eigenvectors.shape), atol=1e-8):
            sorted_eigenvectors = np.real(sorted_eigenvectors)
        else:
            if not self.directed:
                raise ValueError('Complex eigenvectors found in non-directed network!')

        setattr(self, mode + '_eigenvectors', sorted_eigenvectors)

        if verbose:
            print('Diagonalizing finished')

    # TODO: add Gromov hyperbolicity

    def calculate_z_values(self, mode='lap'):
        spectrum = self.get_spectrum(mode)
        seigs = sorted(list(set(spectrum)), key=np.abs)
        if len(seigs) != len(spectrum) and self.verbose:
            print('WARNING:', len(spectrum) - len(seigs), 'repeated eigenvalues discarded')

        if self.verbose:
            print('Computing nearest neighbours...')

        X = np.array([[np.real(x), np.imag(x)] for x in seigs])
        nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)
        nnbs = [seigs[x] for x in indices[:, 1]]
        nnnbs = [seigs[x] for x in indices[:, 2]]
        '''
        nndist = np.array([(nnbs[i] - eigs[i]) for i in range(len(eigs))])
        nnndist = np.array([(nnnbs[i] - eigs[i]) for i in range(len(eigs))])
        '''
        zlist = np.array([(nnbs[i] - seigs[i]) / (nnnbs[i] - seigs[i]) for i in range(len(seigs))])
        zdict = dict(zip(seigs, zlist))

        setattr(self, mode + '_zvalues', zdict)

    def calculate_ipr(self, mode='adj'):
        eigenvectors = self.get_eigenvectors(mode)
        nvecs = eigenvectors.shape[1]
        ipr = np.zeros(nvecs)
        eig_entropy = np.zeros(nvecs)

        for i in range(nvecs):
            ipr[i] = sum([np.abs(v) ** 4 for v in eigenvectors[:, i]])
            # entropy[i] = -np.log(ipr[i]) # erdos entropy (deprecated)
            eig_entropy[i] = entropy(np.array([np.abs(v) ** 2 for v in eigenvectors[:, i]]))

        setattr(self, mode + '_ipr', ipr)
        # self.eigenvector_entropy = eig_entropy / np.log(self.n)

    def _get_lap_spectrum(self, norm=False):
        if not self.directed:
            if norm:
                spectrum = self.get_spectrum('nlap') # could be rwlap as well
            else:
                spectrum = self.get_spectrum('lap')
        else:
            if norm:
                raise NotImplementedError('Normalized Laplacian not implemented for directed networks')
            else:
                spectrum = self.get_spectrum('lap_out')

        return spectrum

    def calculate_thermodynamic_entropy(self, tlist, verbose=False, norm=False):
        spectrum = self._get_lap_spectrum(norm=norm)
        res = [spectral_entropy(spectrum, t, verbose=verbose) for t in tlist]
        return res

    def calculate_free_entropy(self, tlist, norm=False):
        spectrum = self._get_lap_spectrum(norm=norm)
        res = [free_entropy(spectrum, t) for t in tlist]
        return res

    def calculate_q_entropy(self, q, tlist, norm=False):
        spectrum = self._get_lap_spectrum(norm=norm)
        res = [q_entropy(spectrum, t, q=q) for t in tlist]
        return res

    def calculate_estrada_communicability(self):
        adj_spectrum = self.get_spectrum('adj')
        self.estrada_communicability = sum([np.exp(e) for e in adj_spectrum])
        return self.estrada_communicability

    def get_estrada_bipartivity_index(self):
        adj_spectrum = self.get_spectrum('adj')
        esi1 = sum([np.exp(-e) for e in adj_spectrum])
        esi2 = sum([np.exp(e) for e in adj_spectrum])
        self.estrada_bipartivity = esi1 / esi2
        return self.estrada_bipartivity

    def localization_signatures(self, mode='lap'):
        zvals = self.get_z_values(mode)

        mean_cos_phi = np.mean(np.array([np.cos(np.angle(x)) for x in zvals]))
        rvals = [1. / (np.abs(z)) ** 2 for z in zvals]
        mean_inv_r_sq = np.mean(np.array(rvals))

        if self.verbose:
            print('mean cos phi complex:', mean_cos_phi)
            print('mean 1/r^2 real:', mean_inv_r_sq)

        return mean_inv_r_sq, mean_cos_phi

    def construct_lem_embedding(self, dim):
        if self.directed:
            raise Exception('LEM embedding is not implemented for directed graphs')

        A = self.adj
        A = A.asfptype()
        print('Performing spectral decomposition...')
        K = A.shape[0]
        NL = get_norm_laplacian(A)
        DH = get_inv_sqrt_diag_matrix(A)

        start_v = np.ones(K)
        eigvals, eigvecs = eigs(NL, k=dim + 1, which='LR', v0=start_v, maxiter=K * 1000)
        eigvals = np.asarray([np.round(np.real(x), 6) for x in eigvals])

        if np.count_nonzero(eigvals == 1.0) > 1:
            raise Exception('Error while LEM embedding construction: graph is not connected!')
        else:
            vecs = eigvecs.T[1:]
            vec_norms = np.array([np.real(sum([x * x for x in v])) for v in vecs])
            vecs = vecs / vec_norms[:, np.newaxis]
            # explanation: https://jlmelville.github.io/smallvis/spectral.html
            vecs = DH.dot(sp.csr_array(vecs, dtype=float))

            self.lem_emb = vecs
