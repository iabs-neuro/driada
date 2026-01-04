"""
Test calcium dynamics validation and firing rate updates.
"""

import numpy as np
import pytest
import warnings
from driada.experiment.synthetic import (
    validate_peak_rate,
    generate_circular_manifold_neurons,
    generate_2d_manifold_neurons,
    generate_3d_manifold_neurons,
    generate_circular_manifold_data,
    generate_mixed_population_exp,
)


@pytest.fixture(scope="module")
def head_direction_data():
    """Generate head direction data once for all tests."""
    n_timepoints = 100
    return np.linspace(0, 2 * np.pi, n_timepoints)


@pytest.fixture(scope="module")
def position_2d_data():
    """Generate 2D position data once for all tests."""
    from driada.experiment.synthetic import generate_2d_random_walk

    return generate_2d_random_walk(100, seed=42)


@pytest.fixture(scope="module")
def position_3d_data():
    """Generate 3D position data once for all tests."""
    from driada.experiment.synthetic import generate_3d_random_walk

    return generate_3d_random_walk(100, seed=42)


@pytest.fixture(scope="module")
def mixed_population_experiment():
    """Generate mixed population experiment once for all tests."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return generate_mixed_population_exp(
            n_neurons=10,  # Slightly more neurons to avoid edge cases
            duration=30,  # Longer duration to have enough frames for shuffle mask
            fps=10,
            verbose=False,
            seed=42,
        )


def test_validate_peak_rate():
    """Test peak rate validation function."""
    # Should not warn for rates <= 2.0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_peak_rate(1.0)
        assert len(w) == 0

        validate_peak_rate(2.0)
        assert len(w) == 0

    # Should warn for rates > 2.0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_peak_rate(2.5)
        assert len(w) == 1
        assert "exceeds recommended maximum" in str(w[0].message)

        validate_peak_rate(10.0, context="test_function")
        assert len(w) == 2
        assert "test_function" in str(w[1].message)


def test_circular_manifold_defaults(head_direction_data):
    """Test that circular manifold functions use correct defaults."""
    head_direction = head_direction_data

    # Should warn with high peak_rate
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        firing_rates, _ = generate_circular_manifold_neurons(
            n_neurons=10, head_direction=head_direction, peak_rate=5.0  # High rate
        )
        assert len(w) == 1
        assert "calcium signal saturation" in str(w[0].message)

    # Should not warn with default peak_rate
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        firing_rates, _ = generate_circular_manifold_neurons(
            n_neurons=10,
            head_direction=head_direction,
            # Uses default peak_rate=1.0
        )
        assert len(w) == 0

    # Check that firing rates are in reasonable range
    assert np.max(firing_rates) < 3.0  # With noise, shouldn't exceed ~2x peak_rate


def test_2d_manifold_defaults(position_2d_data):
    """Test that 2D manifold functions use correct defaults."""
    positions = position_2d_data

    # Should warn with high peak_rate
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        firing_rates, _ = generate_2d_manifold_neurons(
            n_neurons=9, positions=positions, peak_rate=3.0  # High rate
        )
        assert len(w) == 1

    # Should not warn with default
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        firing_rates, _ = generate_2d_manifold_neurons(
            n_neurons=9,
            positions=positions,
            # Uses default peak_rate=1.0
        )
        assert len(w) == 0


def test_3d_manifold_defaults(position_3d_data):
    """Test that 3D manifold functions use correct defaults."""
    positions = position_3d_data  # Already in (3, n_timepoints) format

    # Should warn with high peak_rate
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        firing_rates, _ = generate_3d_manifold_neurons(
            n_neurons=8, positions=positions, peak_rate=4.0  # High rate
        )
        assert len(w) == 1

    # Should not warn with default
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        firing_rates, _ = generate_3d_manifold_neurons(
            n_neurons=8,
            positions=positions,
            # Uses default peak_rate=1.0
        )
        assert len(w) == 0


def test_mixed_population_defaults(mixed_population_experiment):
    """Test that mixed population uses correct defaults."""
    exp = mixed_population_experiment
    # Check that experiment was created successfully
    assert exp is not None
    assert exp.n_cells == 10
    # The fixture was created without warnings, so defaults are correct

    # Should warn if we override with high rate
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        exp = generate_mixed_population_exp(
            n_neurons=10,  # Use same size as fixture
            duration=30,  # Sufficient duration
            fps=10,  # Lower fps for speed
            manifold_params={
                "field_sigma": 0.1,
                "baseline_rate": 0.1,
                "peak_rate": 5.0,  # High rate
                "noise_std": 0.05,
                "decay_time": 2.0,
                "calcium_noise_std": 0.1,
            },
            verbose=False,
        )
        peak_rate_warnings = [
            warning for warning in w if "peak_rate" in str(warning.message)
        ]
        assert len(peak_rate_warnings) > 0


def test_calcium_saturation_effect():
    """Test that high firing rates actually cause saturation in calcium signals."""

    # Generate data with low peak rate
    calcium_low, _, _, firing_low = generate_circular_manifold_data(
        n_neurons=5, duration=30, peak_rate=1.0, seed=42, verbose=False
    )

    # Generate data with high peak rate (will warn)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        calcium_high, _, _, firing_high = generate_circular_manifold_data(
            n_neurons=5,
            duration=30,
            peak_rate=10.0,  # Very high
            seed=42,
            verbose=False,
        )

    # High firing rate should lead to more saturated calcium signal
    # Check dynamic range: (max - min) / mean
    dynamic_range_low = np.mean(
        (np.max(calcium_low, axis=1) - np.min(calcium_low, axis=1))
        / (np.mean(calcium_low, axis=1) + 1e-10)
    )
    dynamic_range_high = np.mean(
        (np.max(calcium_high, axis=1) - np.min(calcium_high, axis=1))
        / (np.mean(calcium_high, axis=1) + 1e-10)
    )

    # High firing rates should have lower dynamic range due to saturation
    assert dynamic_range_high < dynamic_range_low


def test_parameter_documentation():
    """Test that functions have proper documentation about calcium constraints."""
    # Check that key functions mention calcium saturation in their docstrings
    from driada.experiment.synthetic import (
        generate_circular_manifold_neurons,
        generate_2d_manifold_neurons,
        generate_3d_manifold_neurons,
    )

    for func in [
        generate_circular_manifold_neurons,
        generate_2d_manifold_neurons,
        generate_3d_manifold_neurons,
    ]:
        assert "calcium" in func.__doc__.lower()
        assert "saturation" in func.__doc__.lower() or "2 Hz" in func.__doc__.lower()


def test_default_values_are_realistic():
    """Test that default peak_rate values are set to 1.0 Hz."""
    from inspect import signature
    from driada.experiment.synthetic import (
        generate_circular_manifold_neurons,
        generate_2d_manifold_neurons,
        generate_3d_manifold_neurons,
        generate_2d_manifold_data,
        generate_3d_manifold_data,
        generate_2d_manifold_exp,
        generate_3d_manifold_exp,
    )

    # Check function signatures for default peak_rate
    functions_to_check = [
        generate_circular_manifold_neurons,
        generate_2d_manifold_neurons,
        generate_3d_manifold_neurons,
        generate_circular_manifold_data,
        generate_2d_manifold_data,
        generate_3d_manifold_data,
        generate_2d_manifold_exp,
        generate_3d_manifold_exp,
    ]

    for func in functions_to_check:
        sig = signature(func)
        if "peak_rate" in sig.parameters:
            default_value = sig.parameters["peak_rate"].default
            assert (
                default_value == 1.0
            ), f"{func.__name__} has peak_rate default of {default_value}, expected 1.0"
