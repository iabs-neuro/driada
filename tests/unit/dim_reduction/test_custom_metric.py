"""Test custom metric support in graph construction"""

import numpy as np
from numba import njit
from driada.dim_reduction.graph import ProximityGraph
from driada.dim_reduction.dr_base import m_param_filter


@njit
def manhattan_metric(x, y):
    """Custom Manhattan distance metric - numba compiled"""
    return np.sum(np.abs(x - y))


def test_custom_metric_function():
    """Test that custom metric functions work properly"""
    # Create sample data
    np.random.seed(42)
    data = np.random.randn(3, 20)

    # Test 1: m_param_filter should accept callable metrics
    m_params = {"metric_name": manhattan_metric, "sigma": 1.0}
    filtered_params = m_param_filter(m_params)
    assert "metric_name" in filtered_params
    assert filtered_params["metric_name"] is manhattan_metric
    assert filtered_params["sigma"] == 1.0

    # Test 2: ProximityGraph should work with custom metric
    g_params = {
        "g_method_name": "knn",
        "nn": 5,
        "weighted": False,
        "dist_to_aff": None,
        "max_deleted_nodes": 0.5,
    }

    graph = ProximityGraph(data, m_params, g_params, create_nx_graph=False)

    # Check that graph was constructed
    assert graph.adj is not None
    # After giant component extraction, we may have fewer nodes
    assert graph.adj.shape[0] <= 20
    assert graph.adj.shape[0] == graph.adj.shape[1]  # Square matrix
    assert graph.adj.nnz > 0
    assert graph.metric is manhattan_metric

    print("Test passed: Custom metric functions are supported!")


def test_builtin_metric():
    """Test that built-in metrics still work"""
    np.random.seed(42)
    data = np.random.randn(3, 20)

    # Test with built-in metric
    m_params = {"metric_name": "manhattan", "sigma": 1.0}
    filtered_params = m_param_filter(m_params)
    assert "metric_name" in filtered_params
    assert filtered_params["metric_name"] == "manhattan"

    g_params = {
        "g_method_name": "knn",
        "nn": 5,
        "weighted": False,
        "dist_to_aff": None,
        "max_deleted_nodes": 0.5,
    }

    graph = ProximityGraph(data, m_params, g_params, create_nx_graph=False)
    assert graph.adj is not None
    assert graph.metric == "manhattan"

    print("Test passed: Built-in metrics still work!")


def test_invalid_metric():
    """Test that invalid metrics raise proper errors"""
    # Test with invalid string metric
    m_params = {"metric_name": "invalid_metric", "sigma": 1.0}
    try:
        m_param_filter(m_params)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown metric" in str(e)
        print(f"Test passed: Invalid metric properly rejected with error: {e}")


if __name__ == "__main__":
    test_custom_metric_function()
    test_builtin_metric()
    test_invalid_metric()
    print("\nAll tests passed!")
