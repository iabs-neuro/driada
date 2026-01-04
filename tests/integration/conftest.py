"""Fixtures for integration tests.

This module provides optimized fixtures for integration tests to improve performance
by caching expensive experiment generation operations and providing standard parameter sets.

Key features:
1. Session-scoped fixtures for base experiments (generated once per test session)
2. Function-scoped fixtures that return copies for test isolation
3. Standardized parameter sets (fast/balanced/thorough)
4. Consistent downsampling and optimization settings

Usage:
    def test_my_integration(circular_manifold_exp_fast):
        # Uses fast circular manifold experiment

    @pytest.mark.parametrize("circular_manifold_exp", ["balanced"], indirect=True)
    def test_thorough(circular_manifold_exp):
        # Uses balanced configuration
"""

import pytest
from copy import deepcopy
from driada.experiment.synthetic import (
    generate_circular_manifold_exp,
    generate_2d_manifold_exp,
    generate_mixed_population_exp,
    generate_synthetic_exp,
)


# Cache for expensive experiment generation
_experiment_cache = {}


@pytest.fixture
def integration_test_params(request):
    """Standard parameter sets for integration tests.

    Provides three modes:
    - fast: For CI/development (target: <20s total)
    - balanced: Default mode with reasonable coverage
    - thorough: Comprehensive testing for releases

    Parameters can be overridden per test as needed.
    """
    mode = getattr(request, "param", "fast")

    params = {
        "fast": {
            "n_shuffles_stage1": 10,
            "n_shuffles_stage2": 20,
            "ds": 5,  # Aggressive downsampling
            "n_neurons": 20,
            "duration": 100,
            "verbose": False,
            "allow_mixed_dimensions": True,
            "find_optimal_delays": False,  # Can't use with multifeatures
        },
        "balanced": {
            "n_shuffles_stage1": 25,
            "n_shuffles_stage2": 50,
            "ds": 3,
            "n_neurons": 30,
            "duration": 200,
            "verbose": False,
            "allow_mixed_dimensions": True,
            "find_optimal_delays": False,
        },
        "thorough": {
            "n_shuffles_stage1": 50,
            "n_shuffles_stage2": 100,
            "ds": 2,
            "n_neurons": 40,
            "duration": 400,
            "verbose": False,
            "allow_mixed_dimensions": True,
            "find_optimal_delays": False,
        },
    }
    return params[mode]


# Circular Manifold Experiments
@pytest.fixture(scope="session")
def base_circular_manifold_exp_fast():
    """Fast circular manifold experiment (session-scoped base)."""
    key = "circular_fast"
    if key not in _experiment_cache:
        _experiment_cache[key] = generate_circular_manifold_exp(
            n_neurons=20, duration=100, noise_std=0.1, seed=42
        )
    return _experiment_cache[key]


@pytest.fixture(scope="session")
def base_circular_manifold_exp_balanced():
    """Balanced circular manifold experiment (session-scoped base)."""
    key = "circular_balanced"
    if key not in _experiment_cache:
        _experiment_cache[key] = generate_circular_manifold_exp(
            n_neurons=30, duration=200, noise_std=0.1, seed=42
        )
    return _experiment_cache[key]


@pytest.fixture(scope="session")
def base_circular_manifold_exp_thorough():
    """Thorough circular manifold experiment (session-scoped base)."""
    key = "circular_thorough"
    if key not in _experiment_cache:
        _experiment_cache[key] = generate_circular_manifold_exp(
            n_neurons=40, duration=400, noise_std=0.05, seed=42
        )
    return _experiment_cache[key]


@pytest.fixture
def circular_manifold_exp_fast(base_circular_manifold_exp_fast):
    """Fast circular manifold experiment (function-scoped copy)."""
    return deepcopy(base_circular_manifold_exp_fast)


@pytest.fixture
def circular_manifold_exp_balanced(base_circular_manifold_exp_balanced):
    """Balanced circular manifold experiment (function-scoped copy)."""
    return deepcopy(base_circular_manifold_exp_balanced)


@pytest.fixture
def circular_manifold_exp_thorough(base_circular_manifold_exp_thorough):
    """Thorough circular manifold experiment (function-scoped copy)."""
    return deepcopy(base_circular_manifold_exp_thorough)


@pytest.fixture(params=["fast", "balanced", "thorough"])
def circular_manifold_exp(request):
    """Parametrized circular manifold experiment.

    Use with @pytest.mark.parametrize("circular_manifold_exp", ["fast"], indirect=True)
    to specify a particular configuration.
    """
    fixtures = {
        "fast": request.getfixturevalue("circular_manifold_exp_fast"),
        "balanced": request.getfixturevalue("circular_manifold_exp_balanced"),
        "thorough": request.getfixturevalue("circular_manifold_exp_thorough"),
    }
    return fixtures[request.param]


# 2D Manifold Experiments
@pytest.fixture(scope="session")
def base_2d_manifold_exp_fast():
    """Fast 2D manifold experiment (session-scoped base)."""
    key = "2d_fast"
    if key not in _experiment_cache:
        _experiment_cache[key] = generate_2d_manifold_exp(
            n_neurons=20, duration=100, noise_std=0.2, seed=123
        )
    return _experiment_cache[key]


@pytest.fixture(scope="session")
def base_2d_manifold_exp_balanced():
    """Balanced 2D manifold experiment (session-scoped base)."""
    key = "2d_balanced"
    if key not in _experiment_cache:
        _experiment_cache[key] = generate_2d_manifold_exp(
            n_neurons=25, duration=200, noise_std=0.2, seed=123
        )
    return _experiment_cache[key]


@pytest.fixture(scope="session")
def base_2d_manifold_exp_thorough():
    """Thorough 2D manifold experiment (session-scoped base)."""
    key = "2d_thorough"
    if key not in _experiment_cache:
        _experiment_cache[key] = generate_2d_manifold_exp(
            n_neurons=40, duration=400, noise_std=0.1, seed=123
        )
    return _experiment_cache[key]


@pytest.fixture
def spatial_2d_exp_fast(base_2d_manifold_exp_fast):
    """Fast 2D spatial experiment (function-scoped copy)."""
    return deepcopy(base_2d_manifold_exp_fast)


@pytest.fixture
def spatial_2d_exp_balanced(base_2d_manifold_exp_balanced):
    """Balanced 2D spatial experiment (function-scoped copy)."""
    return deepcopy(base_2d_manifold_exp_balanced)


@pytest.fixture
def spatial_2d_exp_thorough(base_2d_manifold_exp_thorough):
    """Thorough 2D spatial experiment (function-scoped copy)."""
    return deepcopy(base_2d_manifold_exp_thorough)


@pytest.fixture(params=["fast", "balanced", "thorough"])
def spatial_2d_exp(request):
    """Parametrized 2D spatial experiment."""
    fixtures = {
        "fast": request.getfixturevalue("spatial_2d_exp_fast"),
        "balanced": request.getfixturevalue("spatial_2d_exp_balanced"),
        "thorough": request.getfixturevalue("spatial_2d_exp_thorough"),
    }
    return fixtures[request.param]


# Mixed Population Experiments
@pytest.fixture(scope="session")
def base_mixed_population_exp_fast():
    """Fast mixed population experiment (session-scoped base)."""
    key = "mixed_fast"
    if key not in _experiment_cache:
        _experiment_cache[key] = generate_mixed_population_exp(
            n_neurons=30,
            manifold_type="circular",
            manifold_fraction=0.5,
            duration=150,
            seed=42,
        )
    return _experiment_cache[key]


@pytest.fixture(scope="session")
def base_mixed_population_exp_balanced():
    """Balanced mixed population experiment (session-scoped base)."""
    key = "mixed_balanced"
    if key not in _experiment_cache:
        _experiment_cache[key] = generate_mixed_population_exp(
            n_neurons=50,
            manifold_type="circular",
            manifold_fraction=0.5,
            duration=300,
            seed=42,
        )
    return _experiment_cache[key]


@pytest.fixture(scope="session")
def base_mixed_population_exp_thorough():
    """Thorough mixed population experiment (session-scoped base)."""
    key = "mixed_thorough"
    if key not in _experiment_cache:
        _experiment_cache[key] = generate_mixed_population_exp(
            n_neurons=64,
            manifold_type="circular",
            manifold_fraction=0.5,
            duration=500,
            seed=42,
        )
    return _experiment_cache[key]


@pytest.fixture
def mixed_population_exp_fast(base_mixed_population_exp_fast):
    """Fast mixed population experiment (function-scoped copy)."""
    return deepcopy(base_mixed_population_exp_fast)


@pytest.fixture
def mixed_population_exp_balanced(base_mixed_population_exp_balanced):
    """Balanced mixed population experiment (function-scoped copy)."""
    return deepcopy(base_mixed_population_exp_balanced)


@pytest.fixture
def mixed_population_exp_thorough(base_mixed_population_exp_thorough):
    """Thorough mixed population experiment (function-scoped copy)."""
    return deepcopy(base_mixed_population_exp_thorough)


@pytest.fixture(params=["fast", "balanced", "thorough"])
def mixed_population_exp(request):
    """Parametrized mixed population experiment."""
    fixtures = {
        "fast": request.getfixturevalue("mixed_population_exp_fast"),
        "balanced": request.getfixturevalue("mixed_population_exp_balanced"),
        "thorough": request.getfixturevalue("mixed_population_exp_thorough"),
    }
    return fixtures[request.param]


# Custom Experiment Factory
@pytest.fixture
def integration_experiment_factory():
    """Factory for creating custom integration test experiments.

    Provides optimized defaults for integration testing while allowing
    customization as needed.
    """

    def factory(**kwargs):
        # Integration test defaults
        defaults = {
            "n_dfeats": 2,
            "n_cfeats": 2,
            "nneurons": 20,
            "duration": 100,
            "fps": 20,
            "seed": 42,
        }
        # Update with provided parameters
        params = {**defaults, **kwargs}
        return generate_synthetic_exp(**params)

    return factory


# INTENSE Analysis Parameters
@pytest.fixture
def intense_params_fast():
    """Fast INTENSE parameters for integration tests."""
    return {
        "mode": "two_stage",  # Need two_stage for integration test
        "n_shuffles_stage1": 10,
        "n_shuffles_stage2": 20,
        "ds": 5,
        "verbose": False,
        "allow_mixed_dimensions": True,
        "find_optimal_delays": False,
        "enable_parallelization": False,  # Disable for faster single test execution
        "n_jobs": 1,
        "pval_thr": 0.05,  # More lenient threshold for fast tests
        "save_computed_stats": True,  # Save results to experiment for get_significant_neurons
    }


@pytest.fixture
def intense_params_balanced():
    """Balanced INTENSE parameters for integration tests."""
    return {
        "n_shuffles_stage1": 25,
        "n_shuffles_stage2": 50,
        "ds": 3,
        "verbose": False,
        "allow_mixed_dimensions": True,
        "find_optimal_delays": False,
        "enable_parallelization": False,
        "n_jobs": 1,
    }


@pytest.fixture
def intense_params_thorough():
    """Thorough INTENSE parameters for integration tests."""
    return {
        "n_shuffles_stage1": 50,
        "n_shuffles_stage2": 100,
        "ds": 2,
        "verbose": False,
        "allow_mixed_dimensions": True,
        "find_optimal_delays": False,
        "enable_parallelization": False,
        "n_jobs": 1,
    }


@pytest.fixture
def intense_params_parallel():
    """INTENSE parameters with parallelization enabled for performance tests."""
    return {
        "n_shuffles_stage1": 50,
        "n_shuffles_stage2": 100,
        "ds": 2,
        "verbose": False,
        "allow_mixed_dimensions": True,
        "find_optimal_delays": False,
        "enable_parallelization": True,
        "n_jobs": -1,  # Use all available cores
    }


# Memory Efficiency Test Data
@pytest.fixture(scope="session")
def large_experiment_for_memory():
    """Large experiment specifically for memory efficiency testing."""
    key = "large_memory"
    if key not in _experiment_cache:
        _experiment_cache[key] = generate_2d_manifold_exp(
            n_neurons=64, duration=500, seed=42
        )
    return _experiment_cache[key]


@pytest.fixture
def memory_test_exp(large_experiment_for_memory):
    """Memory efficiency test experiment (function-scoped copy)."""
    return deepcopy(large_experiment_for_memory)
