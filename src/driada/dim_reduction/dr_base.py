from pynndescent.distances import named_distances
from typing import Dict, Optional, Any


class DRMethod(object):
    """Dimensionality reduction method configuration.

    Attributes
    ----------
    is_linear : bool
        Whether the method is linear
    requires_graph : bool
        Whether the method requires a proximity graph
    requires_distmat : bool
        Whether the method requires a distance matrix
    nn_based : bool
        Whether the method is neural network based
    handles_disconnected_graphs : bool
        Whether the method can handle disconnected graphs without preprocessing
    default_params : dict
        Default parameters for the embedding method
    default_graph_params : dict or None
        Default graph construction parameters (if requires_graph)
    default_metric_params : dict or None
        Default metric parameters (if requires weights)
        
    Notes
    -----
    Boolean attributes are stored internally but accept 0/1 integer values
    for backward compatibility.    """

    def __init__(
        self,
        is_linear,
        requires_graph,
        requires_distmat,
        nn_based,
        handles_disconnected_graphs=0,
        default_params=None,
        default_graph_params=None,
        default_metric_params=None,
    ):
        """Initialize a DRMethod configuration object.
        
        Parameters
        ----------
        is_linear : int or bool
            Whether the method is linear (1/True) or nonlinear (0/False).
        requires_graph : int or bool
            Whether the method requires a proximity graph (1/True) or not (0/False).
        requires_distmat : int or bool
            Whether the method requires a distance matrix (1/True) or not (0/False).
        nn_based : int or bool
            Whether the method is neural network based (1/True) or not (0/False).
        handles_disconnected_graphs : int or bool, default=0
            Whether the method can handle disconnected graphs without preprocessing.
        default_params : dict or None, default=None
            Default parameters for the embedding method. If None, uses empty dict.
        default_graph_params : dict or None, default=None
            Default graph construction parameters (if requires_graph).
        default_metric_params : dict or None, default=None
            Default metric parameters (if requires weights).        """
        self.is_linear = bool(is_linear)
        self.requires_graph = bool(requires_graph)
        self.requires_distmat = bool(requires_distmat)
        self.nn_based = bool(nn_based)
        self.handles_disconnected_graphs = bool(handles_disconnected_graphs)
        self.default_params = default_params or {}
        self.default_graph_params = default_graph_params
        self.default_metric_params = default_metric_params


# Default graph parameters for graph-based methods
DEFAULT_KNN_GRAPH = {
    "g_method_name": "knn",
    "nn": 15,
    "weighted": 0,
    "max_deleted_nodes": 0.2,
    "dist_to_aff": "hk",
}

DEFAULT_METRIC = {"metric_name": "l2", "sigma": 1.0}

METHODS_DICT = {
    "pca": DRMethod(1, 0, 0, 0, default_params={"dim": 2}),
    "le": DRMethod(
        0,
        1,
        0,
        0,
        default_params={"dim": 2},
        default_graph_params=DEFAULT_KNN_GRAPH,
        default_metric_params=DEFAULT_METRIC,
    ),
    "auto_le": DRMethod(
        0,
        1,
        0,
        0,
        default_params={"dim": 2},
        default_graph_params=DEFAULT_KNN_GRAPH,
        default_metric_params=DEFAULT_METRIC,
    ),
    "dmaps": DRMethod(
        0,
        1,
        0,
        0,
        default_params={"dim": 2, "dm_alpha": 0.5, "dm_t": 1},
        default_graph_params={**DEFAULT_KNN_GRAPH, "weighted": 1},
        default_metric_params=DEFAULT_METRIC,
    ),
    "auto_dmaps": DRMethod(
        0,
        1,
        0,
        0,
        default_params={"dim": 2, "dm_alpha": 0.5, "dm_t": 1},
        default_graph_params=DEFAULT_KNN_GRAPH,
        default_metric_params=DEFAULT_METRIC,
    ),
    "mds": DRMethod(0, 0, 1, 0, default_params={"dim": 2}),
    "isomap": DRMethod(
        0,
        1,
        0,
        0,
        default_params={"dim": 2},
        default_graph_params={**DEFAULT_KNN_GRAPH, "nn": 15},
        default_metric_params=DEFAULT_METRIC,
    ),
    "lle": DRMethod(
        0,
        1,
        0,
        0,
        default_params={"dim": 2},
        default_graph_params={**DEFAULT_KNN_GRAPH, "nn": 10},
        default_metric_params=DEFAULT_METRIC,
    ),
    "hlle": DRMethod(
        0,
        1,
        0,
        0,
        default_params={"dim": 2},
        default_graph_params={**DEFAULT_KNN_GRAPH, "nn": 10},
        default_metric_params=DEFAULT_METRIC,
    ),
    "mvu": DRMethod(
        0,
        1,
        0,
        0,
        default_params={"dim": 2},
        default_graph_params=DEFAULT_KNN_GRAPH,
        default_metric_params=DEFAULT_METRIC,
    ),
    "ae": DRMethod(0, 0, 0, 1, default_params={"dim": 2}),
    "vae": DRMethod(0, 0, 0, 1, default_params={"dim": 2}),
    "flexible_ae": DRMethod(0, 0, 0, 1, default_params={"dim": 2, "architecture": "ae"}),
    "tsne": DRMethod(0, 0, 0, 0, default_params={"dim": 2, "perplexity": 30}),
    "umap": DRMethod(
        0,
        1,
        0,
        0,
        1,  # handles_disconnected_graphs=1
        default_params={"dim": 2, "min_dist": 0.1},
        default_graph_params={**DEFAULT_KNN_GRAPH, "nn": 15},
        default_metric_params=DEFAULT_METRIC,
    ),
}

GRAPH_CONSTRUCTION_METHODS = ["knn", "auto_knn", "eps", "eknn", "umap", "tsne"]

EMBEDDING_CONSTRUCTION_METHODS = [
    "pca",
    "le",
    "auto_le",
    "dmaps",
    "auto_dmaps",
    "mds",
    "isomap",
    "lle",
    "hlle",
    "mvu",
    "ae",
    "vae",
    "flexible_ae",
    "tsne",
    "umap",
]


def m_param_filter(para: Dict[str, Any]) -> Dict[str, Any]:
    """
    This function prunes parameters that are excessive for
    chosen distance matrix construction method.

    Parameters
    ----------
    para : dict
        Dictionary with metric parameters including:
        - metric_name: str or callable - name of metric or custom metric function
        - sigma: float or None - bandwidth parameter
        - p: float - parameter for minkowski metric
        - Other metric-specific parameters

    Returns
    -------
    dict
        Filtered parameters appropriate for the chosen metric
        
    Raises
    ------
    KeyError
        If 'metric_name' key is missing from para dict.
    ValueError
        If metric_name is unknown (not in named_distances, not 'hyperbolic',
        and not callable).
        
    Notes
    -----
    The special metric 'hyperbolic' is supported in addition to the standard
    pynndescent named_distances. Custom callable metrics are also supported.    """
    name = para["metric_name"]
    appr_keys = ["metric_name"]

    if para.get("sigma") is not None:
        appr_keys.append("sigma")

    # Handle different metric types
    if callable(name):
        # Custom metric function - pass through
        pass
    elif name not in named_distances:
        if name == "hyperbolic":
            # Special case for hyperbolic metric
            pass
        else:
            raise ValueError(
                f'Unknown metric "{name}". Metric must be one of {list(named_distances.keys())}, '
                f'"hyperbolic", or a callable custom metric function.'
            )

    # Add metric-specific parameters
    if name == "minkowski" and "p" in para:
        appr_keys.append("p")

    return {key: para[key] for key in appr_keys if key in para}


def g_param_filter(para: Dict[str, Any]) -> Dict[str, Any]:
    """Filter parameters to keep only those relevant for the graph method.
    
    Different graph construction methods require different parameters.
    This function ensures only the appropriate parameters are passed
    to avoid errors or warnings from unused parameters.
    
    Parameters
    ----------
    para : dict
        Dictionary containing all graph construction parameters.
        Must include 'g_method_name' key.
        
    Returns
    -------
    dict
        Filtered dictionary containing only parameters relevant to the
        specified graph construction method.
        
    Raises
    ------
    KeyError
        If 'g_method_name' key is missing from para dict.
        
    Notes
    -----
    Supported graph methods and their specific parameters:
    - 'knn', 'auto_knn', 'umap': requires 'nn' (number of neighbors)
    - 'eps': requires 'eps' (radius) and 'min_density' (minimum graph density)
    - 'eknn': requires 'eps', 'min_density', and 'nn'
    - 'tsne': requires 'perplexity'
    
    All methods support: 'g_method_name', 'max_deleted_nodes', 'weighted',
    'dist_to_aff', 'graph_preprocessing'.
    
    Unknown methods are accepted and will receive only the base parameters.    """
    gmethod = para["g_method_name"]
    appr_keys = ["g_method_name", "max_deleted_nodes", "weighted", "dist_to_aff", "graph_preprocessing"]

    if gmethod in ["knn", "auto_knn", "umap"]:
        appr_keys.extend(["nn"])

    elif gmethod == "eps":
        appr_keys.extend(["eps", "min_density"])

    elif gmethod == "eknn":
        appr_keys.extend(["eps", "min_density", "nn"])

    elif gmethod == "tsne":
        appr_keys.extend(["perplexity"])

    return {key: para[key] for key in appr_keys if key in para}


def e_param_filter(para: Dict[str, Any]) -> Dict[str, Any]:
    """Filter parameters to keep only those relevant for the embedding method.
    
    Different dimensionality reduction methods require different parameters.
    This function ensures only the appropriate parameters are passed to
    avoid errors or warnings from unused parameters.
    
    Parameters
    ----------
    para : dict
        Dictionary containing all embedding parameters.
        Must include 'e_method_name' key.
        
    Returns
    -------
    dict
        Filtered dictionary containing only parameters relevant to the
        specified embedding method.
        
    Raises
    ------
    KeyError
        If 'e_method_name' key is missing from para dict.
        
    Notes
    -----
    All methods support: 'e_method', 'e_method_name', 'dim' (target dimension).
    
    Method-specific parameters:
    - 'umap': adds 'min_dist' (minimum distance in low-dimensional space)
    - 'dmaps', 'auto_dmaps': adds 'dm_alpha' (diffusion maps alpha parameter)
      and 'dm_t' (diffusion time)
      
    Unknown methods are accepted and will receive only the base parameters.    """
    appr_keys = ["e_method", "e_method_name", "dim"]

    if para["e_method_name"] == "umap":
        appr_keys.append("min_dist")

    if para["e_method_name"] in ["dmaps", "auto_dmaps"]:
        appr_keys.append("dm_alpha")
        appr_keys.append("dm_t")

    return {key: para[key] for key in appr_keys if key in para}


def merge_params_with_defaults(
    method_name: str, user_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Dict[str, Any]]:
    """Merge user parameters with method defaults.

    Parameters
    ----------
    method_name : str
        Name of the DR method. Must be one of the keys in METHODS_DICT.
    user_params : dict or None
        User-provided parameters. Can contain 'e_params', 'g_params', 'm_params' keys
        for structured format, or direct parameter values for flat format.

    Returns
    -------
    dict
        Dictionary with 'e_params', 'g_params', 'm_params' keys containing merged parameters.
        
    Raises
    ------
    ValueError
        If method_name is not found in METHODS_DICT.
        
    Notes
    -----
    The function supports two input formats:
    
    1. Structured format with explicit parameter groups:
       {'e_params': {...}, 'g_params': {...}, 'm_params': {...}}
       
    2. Flat format where parameters are auto-distributed:
       - 'n_neighbors' → g_params['nn']
       - 'metric' → m_params['metric_name']
       - 'sigma' → m_params['sigma']
       - 'max_deleted_nodes' → g_params['max_deleted_nodes']
       - All others → e_params
       
    The function also sets graph_preprocessing based on the method's
    handles_disconnected_graphs property:
    - If True: graph_preprocessing = None
    - If False: graph_preprocessing = 'giant_cc'    """
    if method_name not in METHODS_DICT:
        raise ValueError(f"Unknown method: {method_name}")

    method = METHODS_DICT[method_name]

    # Initialize with defaults
    e_params = method.default_params.copy()
    e_params["e_method_name"] = method_name
    e_params["e_method"] = method

    g_params = (
        method.default_graph_params.copy() if method.default_graph_params else None
    )
    
    # Set default graph_preprocessing based on handles_disconnected_graphs property
    if g_params is not None:
        if method.handles_disconnected_graphs:
            g_params.setdefault("graph_preprocessing", None)
        else:
            g_params.setdefault("graph_preprocessing", "giant_cc")
    
    m_params = (
        method.default_metric_params.copy() if method.default_metric_params else None
    )

    if user_params is None:
        return {"e_params": e_params, "g_params": g_params, "m_params": m_params}

    # Handle different input formats
    if (
        "e_params" in user_params
        or "g_params" in user_params
        or "m_params" in user_params
    ):
        # User provided structured parameters
        if "e_params" in user_params and user_params["e_params"]:
            e_params.update(user_params["e_params"])
        if (
            "g_params" in user_params
            and user_params["g_params"]
            and g_params is not None
        ):
            g_params.update(user_params["g_params"])
        if (
            "m_params" in user_params
            and user_params["m_params"]
            and m_params is not None
        ):
            m_params.update(user_params["m_params"])
    else:
        # User provided flat parameters - need to distribute to appropriate dicts
        # CRITICAL: Map ALL common parameter aliases to their correct locations
        for key, value in user_params.items():
            # Graph parameters (g_params)
            if g_params is not None and key in [
                "n_neighbors", "nn", "k",  # All aliases for number of neighbors
                "weighted", "dist_to_aff", "graph_preprocessing",
                "g_method_name", "max_deleted_nodes"
            ]:
                if key in ["n_neighbors", "k"]:  # Map aliases to nn
                    g_params["nn"] = value
                elif key == "nn":  # Direct nn parameter
                    g_params["nn"] = value
                else:  # Other graph params keep their names
                    g_params[key] = value

            # Metric parameters (m_params)
            elif m_params is not None and key in [
                "metric", "metric_name", "sigma"
            ]:
                if key == "metric":  # Map alias to metric_name
                    m_params["metric_name"] = value
                else:  # Direct parameters
                    m_params[key] = value

            # Common embedding parameter aliases
            elif key == "n_components":  # sklearn alias for dim
                e_params["dim"] = value

            # Everything else goes to embedding params
            else:
                e_params[key] = value

    # Always ensure e_method is set
    if "e_method" not in e_params or e_params["e_method"] is None:
        e_params["e_method"] = method

    return {"e_params": e_params, "g_params": g_params, "m_params": m_params}
