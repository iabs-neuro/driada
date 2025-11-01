"""Test public API imports for DRIADA package."""


def test_main_package_import():
    """Test that the main driada package can be imported."""
    import driada

    assert hasattr(driada, "__version__")
    assert isinstance(driada.__version__, str)
    assert len(driada.__version__) > 0


def test_main_exports():
    """Test that main exports are available."""
    import driada

    # Core classes
    assert hasattr(driada, "Experiment")
    assert hasattr(driada, "TimeSeries")
    assert hasattr(driada, "MultiTimeSeries")

    # INTENSE pipelines
    assert hasattr(driada, "compute_cell_feat_significance")
    assert hasattr(driada, "compute_feat_feat_significance")
    assert hasattr(driada, "compute_cell_cell_significance")

    # Information theory
    assert hasattr(driada, "get_mi")
    assert hasattr(driada, "conditional_mi")
    assert hasattr(driada, "interaction_information")

    # Experiment utilities
    assert hasattr(driada, "load_experiment")
    assert hasattr(driada, "save_exp_to_pickle")
    assert hasattr(driada, "load_exp_from_pickle")
    assert hasattr(driada, "generate_synthetic_exp")


def test_submodule_imports():
    """Test that submodules are importable."""
    import driada.intense
    import driada.information
    import driada.experiment
    import driada.utils

    # Check they have __all__ defined
    assert hasattr(driada.intense, "__all__")
    assert hasattr(driada.information, "__all__")
    assert hasattr(driada.experiment, "__all__")
    assert hasattr(driada.utils, "__all__")


def test_intense_module_exports():
    """Test INTENSE module exports."""
    from driada import intense

    # Check key exports
    assert hasattr(intense, "compute_me_stats")
    assert hasattr(intense, "scan_pairs")
    assert hasattr(intense, "disentangle_pair")
    assert hasattr(intense, "plot_disentanglement_heatmap")


def test_information_module_exports():
    """Test information module exports."""
    from driada import information

    # Check key exports
    assert hasattr(information, "get_mi")
    assert hasattr(information, "conditional_mi")
    assert hasattr(information, "interaction_information")
    assert hasattr(information, "entropy_d")
    assert hasattr(information, "mi_gg")


def test_experiment_module_exports():
    """Test experiment module exports."""
    from driada import experiment

    # Check key exports
    assert hasattr(experiment, "Experiment")
    assert hasattr(experiment, "load_experiment")
    assert hasattr(experiment, "generate_synthetic_exp")
    assert hasattr(experiment, "Neuron")


def test_no_internal_leakage():
    """Test that internal implementation details are not exposed."""
    import driada

    # Should not expose internal modules
    public_attrs = [x for x in dir(driada) if not x.startswith("_")]

    # These should not be in the public API
    assert "os" not in public_attrs
    assert "sys" not in public_attrs
    assert "np" not in public_attrs
    assert "numpy" not in public_attrs


def test_convenience_imports():
    """Test that convenience imports work as expected."""
    # Should be able to import key classes/functions directly
    from driada import Experiment, TimeSeries, compute_cell_feat_significance

    # And also from their submodules
    from driada.experiment import Experiment as Exp2
    from driada.information import TimeSeries as TS2
    from driada.intense import compute_cell_feat_significance as compute2

    # They should be the same objects
    assert Experiment is Exp2
    assert TimeSeries is TS2
    assert compute_cell_feat_significance is compute2


def test_dim_reduction_imports():
    """Test dimensionality reduction module imports."""

    # Check DR methods are available
    from driada.dim_reduction import MVData, Embedding

    assert callable(MVData)
    assert callable(Embedding)

    # Check dimensionality estimation functions
    from driada.dimensionality import eff_dim

    assert callable(eff_dim)


def test_network_module_imports():
    """Test network module imports."""
    import driada.network

    # Check main class from net_base
    from driada.network.net_base import Network

    assert callable(Network)

    # Check submodules exist
    assert hasattr(driada.network, "net_base")
    assert hasattr(driada.network, "graph_utils")
    assert hasattr(driada.network, "matrix_utils")


def test_integration_module_imports():
    """Test integration module imports."""

    # Check manifold analysis functions
    from driada.integration import get_functional_organization, compare_embeddings

    assert callable(get_functional_organization)
    assert callable(compare_embeddings)
