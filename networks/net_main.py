from .matrix_utils import *
from .randomization import *
from .drawing import *
from .spectral import *
from .quantum import *

from scipy import linalg as la
from scipy.sparse.linalg import eigs
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy

MATRIX_TYPES = ['adj', 'lap', 'nlap', 'lap_out', 'lap_in']


def check_matrix_type(mode):
    if mode not in MATRIX_TYPES:
        raise ValueError(f'Matrix type {mode} is not in allowed matrix types: {MATRIX_TYPES}')


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
    if is_weighted != bool(weighted):
        raise Exception('Error in network construction, check weights')


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


class Network():
    ''''''

    def __init__(self, a, network_args, pos=None, real_world=0,
                 verbose=0, check_connectivity=1, create_nx_graph=1, node_attrs=None):

        self.directed = network_args['directed']
        if self.directed is None:
            self.directed = not ((a != a.T).nnz == 0)

        self.weighted = network_args['weighted']
        check_directed(self.directed, real_world)

        self.real_world = real_world
        self.verbose = verbose
        self.network_params = network_args
        self.create_nx_graph = create_nx_graph

        for mt in MATRIX_TYPES:
            setattr(self, mt, None)
            setattr(self, mt + '_spectrum', None)
            setattr(self, mt + '_eigenvectors', None)

        self._preprocess_adj_and_pos(a, check_connectivity,
                                     pos=pos, node_attrs=node_attrs)  # each network object has an adjacency matrix from its birth
        self.get_node_degrees()  # each network object has out- and in-degree sequences from its birth

        self.connectivity_checked = check_connectivity

        self.lem_emb = None
        self.zvalues = None
        self.ipr = None
        self.erdos_entropy = None

    def _preprocess_adj_and_pos(self, a, check_connectivity, pos=None, node_attrs=None):
        '''
        This method is closely tied with Network.__init__() and is needed for graph preprocessing.
        Each network is characterized by an input adjacency matrix. This matrix should
        have been already created respecting requirements in self.network_params.
        In other words, it is directed/undirected and weighted/unweighted depending on initial
        settings.
        '''

        '''
        if not sp.issparse(a):
            raise Exception('non-sparse matrix parsed to network constructor!')
        '''
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

                init_g = nx.from_scipy_sparse_array(res, create_using=gtype)
                G = remove_isolates_and_selfloops_from_graph(init_g)  # overkill for safety
                lost_nodes = init_g.number_of_nodes() - G.number_of_nodes()
                lost_edges = init_g.number_of_edges() - G.number_of_edges()
                if lost_nodes + lost_edges != 0:
                    print('%d isolated nodes and %d selfloops removed' % (lost_nodes, lost_edges))

                #############################
                # At this point isolated nodes and selfloops have been removed from graph.
                # An isolated node is a node with zero out-, in- or both degrees, depending on the algorithm.
                # Graph (or DiGraph) 'G' contains all information from sparse matrix 'res' and vice versa.
                #############################

                if self.verbose:
                    print('Obtaining giant component...')
                gc = take_giant_component(G)  # cleared version of a graph
                gc_adj = create_adj_from_graphml(None, graph=gc, gc_checked=1, info=self.real_world,
                                                 directed=directed, weighted=weighted)

                lost_nodes = G.number_of_nodes() - gc.number_of_nodes()
                if lost_nodes > 0 and self.verbose:
                    print('WARNING: %d nodes lost after giant component creation!' % lost_nodes)

                if self.real_world and self.verbose:
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
                gc = nx.from_scipy_sparse_array(a, create_using=gtype)
                init_g = gc
            else:
                gc = None
                raise Exception('not implemented yet')

        # add node positions if provided
        if pos is not None:
            self.pos = {node: pos[node] for node in init_g.nodes() if node in gc.nodes()}
        else:
            self.pos = None

        if node_attrs is not None:
            self.node_attrs = {node: node_attrs[node] for node in init_g.nodes() if node in gc.nodes()}
        else:
            self.node_attrs = None

        # self.graph = nx.from_scipy_sparse_matrix(final, create_using=nx.DiGraph())
        self.graph = gc
        self.adj = gc_adj
        self.n = nx.number_of_nodes(self.graph)

        if self.verbose:
            print('Symmetry index:', get_symmetry_index(gc_adj))

    def randomize(self, rmode='adj'):
        if rmode == 'graph':
            if self.directed:
                g = nx.DiGraph(self.graph)
            else:
                g = nx.Graph(self.graph)

            new_graph = random_rewiring_IOM_preserving(g, r=10)
            a = nx.adjacency_matrix(new_graph)
            new_net = Network(a, self.network_params, pos=self.pos, real_world=0, verbose=0)
            return new_net

        elif rmode == 'adj':
            rand_adj = adj_random_rewiring_iom_preserving(self.adj,
                                                          is_weighted=self.weighted,
                                                          r=10)

            rand_net = Network(rand_adj, self.network_params, pos=self.pos, real_world=0, verbose=0)
            return rand_net

        elif rmode == 'complete':
            rand_adj = random_rewiring_complete_graph(self.adj)
            rand_net = Network(rand_adj, self.network_params, pos=self.pos, real_world=0, verbose=0)
            return rand_net

        else:
            raise ValueError('Unknown randomization method')

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

        hist, bins = np.histogram(deg, bins=max(deg) - min(deg), density=True)
        return hist


    def get_spectrum(self, mode):
        check_matrix_type(mode)
        spectrum = getattr(self, mode + '_spectrum')
        if spectrum is None:
            self.diagonalize(mode=mode)
            spectrum = getattr(self, mode + '_spectrum')

        return spectrum

    def get_eigenvectors(self, mode):
        check_matrix_type(mode)
        eigenvectors = getattr(self, mode + '_eigenvectors')
        if eigenvectors is None:
            self.diagonalize(mode=mode)
            eigenvectors = getattr(self, mode + '_eigenvectors')

        return eigenvectors
    
    
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

        
    def diagonalize(self, mode='lap_out'):
        check_matrix_type(mode)
        if self.verbose:
            print('Preparing for diagonalization...')

        outdeg = self.outdeg
        indeg = self.indeg
        deg = self.deg

        A = self.adj.astype(float)
        n = self.n

        if n != np.count_nonzero(outdeg) and self.connectivity_checked:
            print(n - np.count_nonzero(outdeg), 'nodes without out-edges')
        if n != np.count_nonzero(indeg) and self.connectivity_checked:
            print(n - np.count_nonzero(indeg), 'nodes without in-edges')

        nz = np.count_nonzero(deg)
        if nz != n and self.connectivity_checked:
            print('Graph has', str(n - nz), 'isolated nodes!')
            raise Exception('Graph is not connected!')

        if not self.weighted and not self.directed and not np.allclose(outdeg, indeg):
            raise Exception('out- and in- degrees do not coincide in boolean')

        if mode == 'lap' or mode == 'lap_out':
            matrix = get_laplacian(A)
        elif mode == 'nlap':
            matrix = get_norm_laplacian(A)
        elif mode == 'adj':
            matrix = A.copy()
        else:
            raise Exception(f'diagonalization not implemented for type {mode}')

        if self.verbose:
            print('Performing diagonalization...')

        # raw_eigs, right_eigvecs = np.linalg.eig(matrix.A + R)
        raw_eigs, right_eigvecs = la.eig(matrix.A, right=True)
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

        if self.verbose:
            print('Diagonalization finished')

    def calculate_z_values(self, mode='lap_out'):
        spectrum = self.get_spectrum(mode)

        eigs = sorted(list(set(spectrum)), key=np.abs)
        if len(eigs) != len(spectrum) and self.verbose:
            print('WARNING:', len(spectrum) - len(eigs), 'repeated eigenvalues discarded')

        if self.verbose:
            print('Computing nearest neighbours...')

        X = np.array([[np.real(x), np.imag(x)] for x in eigs])
        nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)
        nnbs = [eigs[x] for x in indices[:, 1]]
        nnnbs = [eigs[x] for x in indices[:, 2]]
        nndist = np.array([(nnbs[i] - eigs[i]) for i in range(len(eigs))])
        nnndist = np.array([(nnnbs[i] - eigs[i]) for i in range(len(eigs))])
        zlist = np.array([(nnbs[i] - eigs[i]) / (nnnbs[i] - eigs[i]) for i in range(len(eigs))])

        self.zvalues = zlist


    def calculate_ipr(self, mode='adj'):
        eigenvectors = self.get_eigenvectors(mode)
        nvecs = eigenvectors.shape[1]
        ipr = np.zeros(nvecs)
        eig_entropy = np.zeros(nvecs)

        for i in range(nvecs):
            ipr[i] = sum([np.abs(v) ** 4 for v in eigenvectors[:, i]])
            # entropy[i] = -np.log(ipr[i]) # erdos entropy (deprecated)
            eig_entropy[i] = entropy(np.array([np.abs(v) ** 2 for v in eigenvectors[:, i]]))

        self.ipr = ipr
        self.eigenvector_entropy = eig_entropy / np.log(self.n)


    def _get_lap_spectrum(self, norm=False):
        if not self.directed:
            if norm:
                spectrum = self.get_spectrum('nlap')
            else:
                spectrum = self.get_spectrum('lap')
        else:
            if norm:
                raise ValueError('not implemented for directed networks')
            else:
                spectrum = self.get_spectrum('lap_out')

        return spectrum

    def calculate_thermodynamic_entropy(self, tlist, verbose=0, norm=0):
        spectrum = self._get_lap_spectrum(norm=norm)
        res = [spectral_entropy(spectrum, t, verbose=verbose) for t in tlist]
        return res

    def calculate_free_energy(self, tlist, norm=0):
        spectrum = self._get_lap_spectrum(norm=norm)
        res = [free_energy(spectrum, t) for t in tlist]
        return res

    def calculate_q_entropy(self, q, tlist, norm=0):
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

    def localization_signatures(self):
        if self.zvalues is None:
            self.calculate_z_values()

        zvals = self.zvalues
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
            vecs = DH.dot(sp.csr_matrix(vecs, dtype=float))

            self.lem_emb = vecs



