from ..utils.plot import create_default_figure
from .matrix_utils import get_laplacian, get_norm_laplacian
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib import cm
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize as color_normalize


def draw_degree_distr(net, mode=None, cumulative=0, survival=1, log_log=0):
    """Draw the degree distribution of a network.

    Visualizes the degree distribution as a histogram or line plot,
    with options for cumulative distributions and log-log scaling.

    Parameters
    ----------
    net : Network
        Network object to analyze.
    mode : {'all', 'in', 'out'} or None, optional
        Which degree type to plot. If None and graph is directed,
        plots all three types. Default is None.
    cumulative : int, optional
        If 1, plot cumulative distribution. Default is 0.
    survival : int, optional
        If 1 and cumulative=1, plot survival function (1-CDF).
        Default is 1.
    log_log : int, optional
        If 1, use log-log scale. Default is 0.

    Returns
    -------
    None
        Displays the plot.

    Notes
    -----
    For directed graphs, shows in-degree, out-degree, and total degree
    distributions in different colors unless mode is specified.
    Log-log plots are useful for identifying power-law distributions.
    In log-log mode, the last element is excluded to avoid log(0).

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use('Agg')  # Use non-interactive backend for testing
    >>> from driada.network import Network
    >>> import networkx as nx
    >>> # Create a simple directed network with a triangle
    >>> edges = [(0, 1), (1, 2), (2, 0)]
    >>> graph = nx.DiGraph(edges)
    >>> net = Network(graph=graph)
    >>> # Draw degree distribution (will show uniform degree of 2)
    >>> draw_degree_distr(net)  # doctest: +SKIP
    >>> # For undirected network
    >>> graph_undir = nx.Graph(edges)
    >>> net_undir = Network(graph=graph_undir)
    >>> draw_degree_distr(net_undir, log_log=1)  # doctest: +SKIP"""
    if not net.directed:
        mode = "all"

    fig, ax = create_default_figure(figsize=(10, 8))
    ax.set_title("Degree distribution", color="white")

    if mode is not None:
        distr = net.get_degree_distr(mode=mode)
        if cumulative:
            if survival:
                distr = 1 - np.cumsum(distr)
            else:
                distr = np.cumsum(distr)

        if log_log:
            (degree,) = ax.loglog(distr[:-1], linewidth=2, c="k", label="degree")
        else:
            (degree,) = ax.plot(distr, linewidth=2, c="k", label="degree")

        ax.legend(handles=[degree], fontsize=16)

    else:
        distr = net.get_degree_distr(mode="all")
        outdistr = net.get_degree_distr(mode="out")
        indistr = net.get_degree_distr(mode="in")
        distrlist = [distr, outdistr, indistr]
        if cumulative:
            if survival:
                distrlist = [1 - np.cumsum(d) for d in distrlist]
            else:
                distrlist = [np.cumsum(d) for d in distrlist]

        if log_log:
            (degree,) = ax.loglog(distrlist[0][:-1], linewidth=2, c="k", label="degree")
            (outdegree,) = ax.loglog(
                distrlist[1][:-1], linewidth=2, c="b", label="outdegree"
            )
            (indegree,) = ax.loglog(
                distrlist[2][:-1], linewidth=2, c="r", label="indegree"
            )
        else:
            (degree,) = ax.plot(distrlist[0], linewidth=2, c="k", label="degree")
            (outdegree,) = ax.plot(distrlist[1], linewidth=2, c="b", label="outdegree")
            (indegree,) = ax.plot(distrlist[2], linewidth=2, c="r", label="indegree")

        ax.legend(handles=[degree, outdegree, indegree], fontsize=16)


def draw_spectrum(net, mode="adj", ax=None, colors=None, cmap="plasma", nbins=None):
    """Visualize the eigenvalue spectrum of a network matrix.

    For directed graphs, plots eigenvalues in the complex plane.
    For undirected graphs, shows a histogram of real eigenvalues.

    Parameters
    ----------
    net : Network
        Network object to analyze.
    mode : {'adj', 'lap', 'nlap'}, optional
        Matrix type: 'adj' for adjacency, 'lap' for Laplacian,
        'nlap' for normalized Laplacian. Default is 'adj'.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure. Default is None.
    colors : array-like, optional
        Colors for scatter plot points (directed graphs only).
        Default is None.
    cmap : str, optional
        Colormap name. Default is 'plasma'.
    nbins : int, optional
        Number of histogram bins (undirected graphs only).
        Default is len(spectrum)/10.

    Returns
    -------
    None
        Modifies the provided axes or displays a new plot.

    Notes
    -----
    The spectrum provides insights into network properties:
    - Largest eigenvalue relates to network connectivity
    - Spectral gap indicates mixing time and robustness
    - Complex eigenvalues (directed) indicate cyclic structure

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use('Agg')  # Use non-interactive backend for testing
    >>> from driada.network import Network
    >>> import networkx as nx
    >>> # Create a small cycle graph
    >>> graph = nx.cycle_graph(5)
    >>> net = Network(graph=graph)
    >>> # Draw adjacency matrix spectrum
    >>> draw_spectrum(net, mode='adj')  # doctest: +SKIP
    >>> # Draw Laplacian spectrum - eigenvalues should be non-negative
    >>> draw_spectrum(net, mode='lap')  # doctest: +SKIP"""
    spectrum = net.get_spectrum(mode)
    data = np.array(sorted(list(set(spectrum)), key=np.abs))

    if ax is None:
        fig, ax = create_default_figure(12, 10)

    if net.directed:
        ax.scatter(data.real, data.imag, cmap=cmap, c=colors)
    else:
        if nbins is None:
            nbins = max(1, len(spectrum) // 10)
        ax.hist(data.real, bins=nbins)


def get_vector_coloring(vec, cmap="plasma"):
    """Create color mapping for vector values.

    Maps numerical values to colors using matplotlib colormap,
    normalizing to [0, 1] range.

    Parameters
    ----------
    vec : array-like
        Vector of numerical values to map to colors.
    cmap : str, optional
        Matplotlib colormap name. Default is 'plasma'.

    Returns
    -------
    numpy.ndarray
        Array of RGBA color values, shape (n, 4).
        
    Raises
    ------
    ValueError
        If all values in vec are identical (no variation to map).

    Examples
    --------
    >>> import numpy as np
    >>> values = [0.1, 0.5, 0.9, 0.3]
    >>> colors = get_vector_coloring(values, cmap='viridis')
    >>> colors.shape
    (4, 4)
    >>> # Colors are RGBA values in range [0, 1]
    >>> assert np.all(colors >= 0) and np.all(colors <= 1)
    >>> # Test with uniform values - should raise error
    >>> try:
    ...     get_vector_coloring([1.0, 1.0, 1.0])
    ... except ValueError as e:
    ...     print("Error:", str(e))
    Error: All values in vector are identical, cannot create color mapping"""
    cmap = plt.get_cmap(cmap)
    vec = np.array(vec).ravel()
    vec_min = np.min(vec)
    vec_max = np.max(vec)
    
    if vec_max == vec_min:
        raise ValueError("All values in vector are identical, cannot create color mapping")
    
    colors = cmap((vec - vec_min) / (vec_max - vec_min))
    return colors


def draw_eigenvectors(
    net,
    left_ind,
    right_ind,
    mode="adj",
    nodesize=None,
    cmap="plasma",
    draw_edges=True,
    edge_options={},
):
    """Draw network nodes colored by eigenvector components.
    
    Visualizes a network with nodes colored according to the values of
    eigenvectors in the specified index range. Creates a grid of subplots,
    one for each eigenvector from left_ind to right_ind (inclusive).
    
    Parameters
    ----------
    net : Network
        Network object to visualize.
    left_ind : int
        Starting index of eigenvectors to visualize (inclusive).
    right_ind : int
        Ending index of eigenvectors to visualize (inclusive).
    mode : str, optional
        Matrix mode for eigendecomposition. Options: 'adj', 'lap', 'nlap'.
        Default is 'adj'.
    nodesize : float or None, optional
        Size of nodes in the visualization. If None, uses default size.
        Default is None.
    cmap : str or matplotlib colormap, optional
        Colormap for coloring nodes. Default is 'plasma'.
    draw_edges : bool, optional
        Whether to draw edges. Default is True.
    edge_options : dict, optional
        Additional options for edge drawing (passed to networkx).
        Default is empty dict.
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the eigenvector visualization.
        
    Notes
    -----
    The function creates a grid of subplots arranged to fit all requested
    eigenvectors. Each subplot shows the network with nodes colored by
    the corresponding eigenvector's components. The subplot title shows
    the eigenvector index and its eigenvalue.
    
    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use('Agg')  # Use non-interactive backend for testing
    >>> from driada.network import Network
    >>> import networkx as nx
    >>> # Create a graph with interesting spectral properties
    >>> graph = nx.cycle_graph(8)
    >>> net = Network(graph=graph)
    >>> # Visualize first 4 eigenvectors of adjacency matrix
    >>> draw_eigenvectors(net, 0, 3, mode='adj')  # doctest: +SKIP
    >>> # Visualize Laplacian eigenvectors (Fiedler vector is at index 1)
    >>> draw_eigenvectors(net, 1, 2, mode='lap')  # doctest: +SKIP
    """
    spectrum = net.get_spectrum(mode)
    eigenvectors = net.get_eigenvectors(mode)

    vecs = eigenvectors[:, left_ind : right_ind + 1]
    # vecs = np.abs(net.eigenvectors[:, left_ind: right_ind+1])
    eigvals = np.real(spectrum[left_ind : right_ind + 1])

    npics = vecs.shape[1]
    pics_in_a_row = int(np.ceil(np.sqrt(npics)))
    pics_in_a_col = int(np.ceil(1.0 * npics / pics_in_a_row))
    fig, axs = plt.subplots(
        nrows=pics_in_a_col,
        ncols=pics_in_a_row,
        figsize=(8 * pics_in_a_row, 8 * pics_in_a_col),
    )
    for ax in axs.ravel():
        ax.set_axis_off()
    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, hspace=0.2, wspace=0.1
    )

    if net.pos is None:
        pos = nx.layout.spring_layout(net.graph)
        # pos = nx.drawing.layout.circular_layout(net.graph)
    else:
        pos = net.pos

    if nodesize is None:
        nodesize = np.sqrt(net.scaled_outdeg) * 500 + 50

    for i in range(npics):
        vec = vecs[:, i]
        ax = fig.add_subplot(pics_in_a_col, pics_in_a_row, i + 1)

        text = "eigenvector " + str(i + 1) + " lambda " + str(np.round(eigvals[i], 3))
        ax.set_title(text)

        ncolors = get_vector_coloring(vec)
        options = {
            "node_color": ncolors,
            "node_size": nodesize,
            "cmap": mpl.colormaps[cmap],
        }

        nx.draw_networkx_nodes(net.graph, pos, ax=ax, **options)

        if draw_edges:
            nx.draw_networkx_edges(net.graph, pos, **edge_options)
        # pc, = mpl.collections.PatchCollection(nodes, cmap = options['cmap'])
        # pc.set_array(edge_colors)

        cmappable = ScalarMappable(color_normalize(0, 1), cmap=cmap)
        fig.colorbar(cmappable, ax=ax)
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()
    return fig


def draw_net(net, colors=None, nodesize=None, ax=None):
    """Visualize a network graph with customizable node properties.

    Parameters
    ----------
    net : Network
        Network object to visualize.
    colors : array-like, optional
        Node colors. Can be a single color or array of values
        to be mapped to colors. Default is None.
    nodesize : array-like, optional
        Node sizes. If None, sizes are based on node out-degree.
        Default is None.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure. Default is None.

    Returns
    -------
    None
        Displays the network visualization.

    Notes
    -----
    If no node positions are stored in the network, uses spring
    layout for automatic positioning. Node sizes by default are
    proportional to the square root of scaled out-degree.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use('Agg')  # Use non-interactive backend for testing
    >>> from driada.network import Network
    >>> import networkx as nx
    >>> import numpy as np
    >>> # Create a small network with a path
    >>> edges = [(0, 1), (1, 2)]
    >>> graph = nx.Graph(edges)
    >>> net = Network(graph=graph, create_nx_graph=True)
    >>> # Draw network with default settings
    >>> draw_net(net)  # doctest: +SKIP
    >>> # Draw with custom colors based on node index
    >>> colors = [0, 0.5, 1]  # Three nodes with gradient colors
    >>> draw_net(net, colors=colors)  # doctest: +SKIP"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 12))

    if net.pos is None:
        print("Node positions not found, auto layout was constructed")
        pos = nx.layout.spring_layout(net.graph)
        # pos = nx.drawing.layout.circular_layout(net.graph)
    else:
        pos = net.pos

    if nodesize is None:
        nodesize = np.sqrt(net.scaled_outdeg) * 100 + 10

    node_options = {
        "node_size": nodesize,
        "cmap": cm.get_cmap("Spectral"),
    }
    edge_options = {}

    nx.draw_networkx_nodes(
        net.graph, pos, node_color=colors, ax=ax, **node_options
    )
    nx.draw_networkx_edges(net.graph, pos, ax=ax, **edge_options)

    plt.show()


def show_mat(net, dtype=None, mode="adj", ax=None):
    """Display a network matrix as a heatmap.

    Parameters
    ----------
    net : Network
        Network object containing the matrix.
    dtype : numpy.dtype, optional
        Data type to cast matrix to before display.
        Useful for binary visualization. Default is None.
    mode : str, optional
        Matrix type to display:
        - 'adj': adjacency matrix
        - 'lap'/'lap_out': Laplacian matrix
        - 'nlap': normalized Laplacian
        Default is 'adj'.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure. Default is None.

    Returns
    -------
    None
        Displays the matrix visualization.

    Notes
    -----
    If the requested matrix hasn't been computed yet, it will be
    generated on demand. Sparse matrices are converted to dense
    for visualization.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use('Agg')  # Use non-interactive backend for testing
    >>> from driada.network import Network
    >>> import networkx as nx
    >>> import numpy as np
    >>> # Create a small cycle graph
    >>> edges = [(0, 1), (1, 2), (2, 0)]
    >>> graph = nx.Graph(edges)
    >>> net = Network(graph=graph)
    >>> # Show adjacency matrix as binary (shows connections)
    >>> show_mat(net, dtype=bool)  # doctest: +SKIP
    >>> # Show Laplacian matrix (degree matrix minus adjacency)
    >>> show_mat(net, mode='lap')  # doctest: +SKIP
    >>> # For a directed graph
    >>> digraph = nx.DiGraph(edges)
    >>> net_dir = Network(graph=digraph)
    >>> show_mat(net_dir, mode='adj')  # doctest: +SKIP"""
    mat = getattr(net, mode)
    if mat is None:
        if mode in ["lap", "lap_out"]:
            mat = get_laplacian(net.adj)
        elif mode == "nlap":
            mat = get_norm_laplacian(net.adj)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Convert to dense array if sparse
    if hasattr(mat, 'toarray'):
        mat_dense = mat.toarray()
    else:
        mat_dense = np.asarray(mat)
    
    if dtype is not None:
        ax.matshow(mat_dense.astype(dtype))
    else:
        ax.matshow(mat_dense)


def plot_lem_embedding(net, ndim, colors=None):
    """Plot Laplacian Eigenmaps embedding of the network.

    Computes a low-dimensional embedding using Laplacian Eigenmaps
    and visualizes nodes in the embedded space.

    Parameters
    ----------
    net : Network
        Network object to embed.
    ndim : int
        Number of dimensions for embedding (2 or 3).
    colors : array-like, optional
        Node colors for visualization. Default is None.

    Returns
    -------
    None
        Displays the embedding plot.

    Notes
    -----
    Laplacian Eigenmaps use the eigenvectors of the graph Laplacian
    corresponding to the smallest non-zero eigenvalues to embed
    nodes in a low-dimensional space while preserving local structure.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use('Agg')  # Use non-interactive backend for testing
    >>> from driada.network import Network
    >>> import networkx as nx
    >>> # Create a small network that can be well-embedded
    >>> graph = nx.cycle_graph(6)
    >>> net = Network(graph=graph)
    >>> # Create 2D Laplacian Eigenmaps embedding
    >>> plot_lem_embedding(net, ndim=2)  # doctest: +SKIP
    >>> # Create 3D embedding for more complex visualization
    >>> graph_3d = nx.complete_graph(5)
    >>> net_3d = Network(graph=graph_3d)
    >>> plot_lem_embedding(net_3d, ndim=3)  # doctest: +SKIP"""

    if net.lem_emb is None:
        net.construct_lem_embedding(ndim)

    if colors is None:
        colors = range(net.lem_emb.shape[1])

    psize = 10
    # Handle both sparse and dense arrays
    if hasattr(net.lem_emb, 'toarray'):
        data = net.lem_emb.toarray()
    else:
        data = net.lem_emb
    pairs = list(combinations(np.arange(ndim), 2))
    npics = len(pairs)
    pics_in_a_row = int(np.ceil(np.sqrt(npics)))
    pics_in_a_col = int(np.ceil(1.0 * npics / pics_in_a_row))

    fig = plt.figure(figsize=(20, 20))
    fig.suptitle("Projections")
    for i in range(len(pairs)):
        ax = fig.add_subplot(pics_in_a_row, pics_in_a_col, i + 1)
        i1, i2 = pairs[i]
        scatter = ax.scatter(data[i1, :], data[i2, :], c=colors, s=psize)

        ax.legend(
            *scatter.legend_elements(), loc="upper left", title="Classes"
        )

        ax.text(
            min(data[i1, :]),
            min(data[i2, :]),
            "axes " + str(i1 + 1) + " vs." + str(i2 + 1),
            bbox={"facecolor": "red", "alpha": 0.5, "pad": 10},
        )
        # ax.add_artist(legend)
