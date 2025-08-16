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
    """

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
        self.is_linear = is_linear
        self.requires_graph = requires_graph
        self.requires_distmat = requires_distmat
        self.nn_based = nn_based
        self.handles_disconnected_graphs = handles_disconnected_graphs
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
    """
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


def g_param_filter(para):
    """
    This function prunes parameters that are excessive for
    chosen graph construction method
    """
    gmethod = para["g_method_name"]
    appr_keys = ["g_method_name", "max_deleted_nodes", "weighted", "dist_to_aff", "graph_preprocessing"]

    if gmethod in ["knn", "auto_knn", "umap"]:
        appr_keys.extend(["nn"])

    elif gmethod == "eps":
        appr_keys.extend(["eps", "eps_min"])

    elif gmethod == "eknn":
        appr_keys.extend(["eps", "eps_min", "nn"])

    elif gmethod == "tsne":
        appr_keys.extend(["perplexity"])

    return {key: para[key] for key in appr_keys}


def e_param_filter(para):
    """
    This function prunes parameters that are excessive for the
    chosen embedding construction method
    """

    appr_keys = ["e_method", "e_method_name", "dim"]

    if para["e_method_name"] == "umap":
        appr_keys.append("min_dist")

    if para["e_method_name"] in ["dmaps", "auto_dmaps"]:
        appr_keys.append("dm_alpha")
        appr_keys.append("dm_t")

    return {key: para[key] for key in appr_keys}


def merge_params_with_defaults(
    method_name: str, user_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Dict[str, Any]]:
    """Merge user parameters with method defaults.

    Parameters
    ----------
    method_name : str
        Name of the DR method
    user_params : dict or None
        User-provided parameters. Can contain 'e_params', 'g_params', 'm_params' keys
        or direct parameter values which will be treated as embedding parameters.

    Returns
    -------
    dict
        Dictionary with 'e_params', 'g_params', 'm_params' keys containing merged parameters
    """
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
        for key, value in user_params.items():
            if key == "n_neighbors" and g_params is not None:
                # Map n_neighbors to nn in graph params
                g_params["nn"] = value
            elif key == "metric" and m_params is not None:
                # Map metric to metric_name in metric params
                m_params["metric_name"] = value
            elif key == "sigma" and m_params is not None:
                m_params["sigma"] = value
            elif key == "max_deleted_nodes" and g_params is not None:
                g_params["max_deleted_nodes"] = value
            else:
                # All other params go to embedding params
                e_params[key] = value

    # Always ensure e_method is set
    if "e_method" not in e_params or e_params["e_method"] is None:
        e_params["e_method"] = method

    return {"e_params": e_params, "g_params": g_params, "m_params": m_params}
