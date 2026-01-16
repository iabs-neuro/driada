"""
Example of Representational Similarity Analysis (RSA) with DRIADA.

This example demonstrates:
1. Computing RDMs from stimulus-selective neural populations
2. Comparing representations between brain regions
3. Statistical testing with bootstrap methods
4. Spatial selectivity analysis with place cells
"""

import numpy as np
import matplotlib.pyplot as plt
import driada
from driada import rsa
from driada.dim_reduction.data import MVData
from driada.experiment.synthetic import generate_tuned_selectivity_exp


def create_stimulus_labels_from_events(exp, event_names):
    """
    Convert multiple binary event features to categorical stimulus labels.

    Parameters
    ----------
    exp : Experiment
        DRIADA experiment with discrete event features
    event_names : list of str
        Event feature names (e.g., ["event_0", "event_1", "event_2", "event_3"])

    Returns
    -------
    labels : np.ndarray
        Categorical labels (0, 1, 2, ...) for each timepoint.
        Timepoints with no event are labeled -1.

    Notes
    -----
    If events overlap, the last event in the list takes precedence.
    """
    n_timepoints = exp.calcium.scdata.shape[1]
    labels = np.full(n_timepoints, -1, dtype=int)

    # Assign labels based on which event is active
    for idx, event_name in enumerate(event_names):
        event_data = exp.dynamic_features[event_name].data
        labels[event_data > 0] = idx

    return labels


def example_1_stimulus_conditions():
    """Example 1: RDM from stimulus-selective neurons."""
    print("\n=== Example 1: RSA with Stimulus Conditions ===")

    # Define population: neurons selective to 4 different stimuli
    population = [
        {
            "name": "stimulus_selective",
            "count": 45,
            "features": ["event_0", "event_1", "event_2", "event_3"],
            "combination": "or",  # Respond to any of the 4 stimuli
        },
        {
            "name": "nonselective",
            "count": 5,
            "features": [],  # Background activity only
        },
    ]

    # Generate synthetic data with event features as stimuli
    print("Generating stimulus-selective neurons...")
    exp = generate_tuned_selectivity_exp(
        population=population,
        n_discrete_features=4,  # Creates event_0, event_1, event_2, event_3
        duration=300,
        event_active_fraction=0.15,
        event_avg_duration=1.0,
        baseline_rate=0.05,
        peak_rate=2.0,
        seed=42,
        verbose=False,
    )

    # Convert binary event features to categorical stimulus labels
    print("Computing RDM...")
    stimulus_labels = create_stimulus_labels_from_events(
        exp, ["event_0", "event_1", "event_2", "event_3"]
    )

    # Compute RDM across stimulus conditions
    # Filter to only include timepoints where events are active
    valid_mask = stimulus_labels >= 0
    rdm, labels = rsa.compute_rdm_unified(
        exp.calcium.scdata[:, valid_mask].T, items=stimulus_labels[valid_mask]
    )

    print(f"RDM shape: {rdm.shape}")
    print(f"Stimulus conditions: {labels}")

    # Visualize RDM
    label_names = ["Stim A", "Stim B", "Stim C", "Stim D"]
    fig = rsa.plot_rdm(
        rdm,
        labels=label_names[: len(labels)],
        title="Neural RDM - Stimulus Conditions",
        show_values=True,
    )
    plt.savefig("rsa_example_1.png")
    plt.close()

    return exp, rdm, labels


def example_2_compare_regions():
    """Example 2: Comparing representations between brain regions."""
    print("\n=== Example 2: Comparing Representations ===")

    # Generate two related neural populations (e.g., V1 and V2)
    np.random.seed(42)
    n_items = 20
    n_neurons_v1 = 100
    n_neurons_v2 = 150

    # Create base patterns that both regions respond to
    base_patterns = np.random.randn(n_items, 50)

    # V1: Direct representation with noise
    v1_data = base_patterns @ np.random.randn(50, n_neurons_v1)
    v1_data += 0.2 * np.random.randn(n_items, n_neurons_v1)

    # V2: Transformed representation with noise
    transform = np.random.randn(50, 50)
    v2_data = (base_patterns @ transform) @ np.random.randn(50, n_neurons_v2)
    v2_data += 0.2 * np.random.randn(n_items, n_neurons_v2)

    # Compare representations directly
    print("Comparing V1 and V2 representations...")
    similarity = rsa.rsa_compare(v1_data, v2_data)
    print(f"V1-V2 similarity (Spearman): {similarity:.3f}")

    # Try different distance metrics
    print("\nTrying different distance metrics:")
    for metric in ["correlation", "euclidean", "cosine"]:
        sim = rsa.rsa_compare(v1_data, v2_data, metric=metric)
        print(f"  {metric}: {sim:.3f}")

    # Try different comparison methods
    print("\nTrying different comparison methods:")
    for comparison in ["spearman", "pearson", "kendall"]:
        sim = rsa.rsa_compare(v1_data, v2_data, comparison=comparison)
        print(f"  {comparison}: {sim:.3f}")

    # Visualize both RDMs
    rdm1 = rsa.compute_rdm(v1_data)
    rdm2 = rsa.compute_rdm(v2_data)

    fig = rsa.plot_rdm_comparison(
        [rdm1, rdm2], titles=["V1 Representation", "V2 Representation"]
    )
    plt.savefig("rsa_example_2_simplified.png")
    plt.close()

    return similarity


def example_3_compare_experiments():
    """Example 3: Comparing experiments from different sessions."""
    print("\n=== Example 3: Comparing Experiments ===")

    # Define population with stimulus-selective neurons
    # Use shared selectivity with OR combination so neurons respond to multiple stimuli
    population = [
        {
            "name": "stimulus_selective",
            "count": 35,
            "features": ["event_0", "event_1", "event_2"],
            "combination": "or",
        },
        {"name": "nonselective", "count": 5, "features": []},
    ]

    # Session 1
    print("Generating session 1...")
    exp1 = generate_tuned_selectivity_exp(
        population=population,
        n_discrete_features=3,
        duration=240,
        event_active_fraction=0.20,
        event_avg_duration=1.5,
        baseline_rate=0.1,
        peak_rate=2.0,
        seed=42,
        verbose=False,
    )

    # Session 2 (different seed = different noise, but same structure)
    print("Generating session 2...")
    exp2 = generate_tuned_selectivity_exp(
        population=population,
        n_discrete_features=3,
        duration=240,
        event_active_fraction=0.20,
        event_avg_duration=1.5,
        baseline_rate=0.1,
        peak_rate=2.0,
        seed=123,
        verbose=False,
    )

    # Compare experiments using categorical stimulus labels
    print("Comparing two experiments...")
    stim_labels_1 = create_stimulus_labels_from_events(
        exp1, ["event_0", "event_1", "event_2"]
    )
    stim_labels_2 = create_stimulus_labels_from_events(
        exp2, ["event_0", "event_1", "event_2"]
    )

    # Filter to valid timepoints
    valid_1 = stim_labels_1 >= 0
    valid_2 = stim_labels_2 >= 0

    # Compute RDMs using euclidean distance
    rdm1, _ = rsa.compute_rdm_unified(
        exp1.calcium.scdata[:, valid_1].T, items=stim_labels_1[valid_1], metric="euclidean"
    )
    rdm2, _ = rsa.compute_rdm_unified(
        exp2.calcium.scdata[:, valid_2].T, items=stim_labels_2[valid_2], metric="euclidean"
    )

    similarity = rsa.compare_rdms(rdm1, rdm2, method="spearman")
    print(f"Experiment similarity: {similarity:.3f}")

    # Visualize the RDMs
    fig = rsa.plot_rdm_comparison(
        [rdm1, rdm2],
        labels=["Stim A", "Stim B", "Stim C"],
        titles=["Session 1", "Session 2"],
    )
    plt.savefig("rsa_example_3.png")
    plt.close()

    return similarity


def example_4_spatial_selectivity():
    """Example 4: RDM from spatial selectivity."""
    print("\n=== Example 4: RSA with Spatial Tuning ===")

    # Define population with spatial selectivity
    population = [
        {"name": "place_cells", "count": 50, "features": ["position_2d"]},
        {"name": "head_direction_cells", "count": 10, "features": ["head_direction"]},
        {"name": "nonselective", "count": 4, "features": []},
    ]

    # Generate neurons with spatial tuning
    print("Generating spatially-selective neurons...")
    exp = generate_tuned_selectivity_exp(
        population=population,
        duration=600,
        baseline_rate=0.1,
        peak_rate=2.0,
        seed=42,
        verbose=False,
    )

    # Bin spatial positions into discrete regions
    print("Binning spatial positions...")
    position_data = exp.dynamic_features["position_2d"].data
    x, y = position_data[0], position_data[1]

    # Create 3x3 spatial grid (fewer bins to avoid sparse regions)
    x_bins = np.digitize(x, np.linspace(x.min(), x.max(), 4))
    y_bins = np.digitize(y, np.linspace(y.min(), y.max(), 4))
    spatial_bins = x_bins * 3 + y_bins

    # Compute RDM using euclidean distance (more robust than correlation)
    print("Computing spatial RDM...")
    rdm, labels = rsa.compute_rdm_unified(
        exp.calcium.scdata.T, items=spatial_bins, metric="euclidean"
    )

    print(f"RDM shape: {rdm.shape}")
    print(f"Unique spatial bins: {len(labels)}")

    # Visualize spatial RDM (no dendrogram to avoid NaN issues)
    fig = rsa.plot_rdm(
        rdm,
        labels=[f"Bin {i}" for i in labels],
        title="Neural RDM - Spatial Locations",
        dendrogram_ratio=0,
    )
    plt.savefig("rsa_example_4.png")
    plt.close()

    return exp, rdm, labels


def example_5_bootstrap_testing():
    """Example 5: Statistical testing with bootstrap methods."""
    print("\n=== Example 5: Bootstrap Statistical Testing ===")

    # Define population with stimulus selectivity
    population = [
        {
            "name": "stimulus_selective",
            "count": 45,
            "features": ["event_0", "event_1", "event_2", "event_3"],
            "combination": "or",
        },
        {"name": "background", "count": 5, "features": []},
    ]

    # Generate two populations with different noise levels
    print("Generating two neural populations...")
    exp1 = generate_tuned_selectivity_exp(
        population=population,
        n_discrete_features=4,
        duration=300,
        seed=42,
        verbose=False,
    )

    exp2 = generate_tuned_selectivity_exp(
        population=population,
        n_discrete_features=4,
        duration=300,
        seed=123,
        calcium_noise=0.05,  # More noise in population 2
        verbose=False,
    )

    # Create stimulus labels from events
    print("Creating stimulus labels...")
    stim_labels_1 = create_stimulus_labels_from_events(
        exp1, ["event_0", "event_1", "event_2", "event_3"]
    )
    stim_labels_2 = create_stimulus_labels_from_events(
        exp2, ["event_0", "event_1", "event_2", "event_3"]
    )

    # Compute RDMs (filter to only timepoints with events)
    valid_mask_1 = stim_labels_1 >= 0
    valid_mask_2 = stim_labels_2 >= 0

    rdm1, labels1 = rsa.compute_rdm_unified(
        exp1.calcium.scdata[:, valid_mask_1].T,
        items=stim_labels_1[valid_mask_1],
        metric="euclidean",
    )

    rdm2, labels2 = rsa.compute_rdm_unified(
        exp2.calcium.scdata[:, valid_mask_2].T,
        items=stim_labels_2[valid_mask_2],
        metric="euclidean",
    )

    # Compare RDMs
    print("\nComparing RDMs with multiple methods:")
    for method in ["spearman", "pearson", "kendall"]:
        similarity = rsa.compare_rdms(rdm1, rdm2, method=method)
        print(f"  {method}: {similarity:.3f}")

    # Bootstrap significance test (pass data without transpose)
    print("\nRunning bootstrap significance test...")
    bootstrap_results = rsa.bootstrap_rdm_comparison(
        exp1.calcium.scdata[:, valid_mask_1],
        exp2.calcium.scdata[:, valid_mask_2],
        stim_labels_1[valid_mask_1],
        stim_labels_2[valid_mask_2],
        metric="euclidean",
        n_bootstrap=100,
        random_state=42,
    )

    print(f"Bootstrap p-value: {bootstrap_results['p_value']:.3f}")
    print(
        f"95% CI: [{bootstrap_results['ci_lower']:.3f}, {bootstrap_results['ci_upper']:.3f}]"
    )

    # Visualize both RDMs
    fig = rsa.plot_rdm_comparison(
        [rdm1, rdm2], labels=list(labels1), titles=["Population 1", "Population 2"]
    )
    plt.savefig("rsa_example_5.png")
    plt.close()

    return rdm1, rdm2, bootstrap_results["observed"]


def example_6_mvdata_integration():
    """Example 6: Working with MVData objects."""
    print("\n=== Example 6: MVData Integration ===")

    # Create MVData object with known structure
    n_features = 100
    n_timepoints = 1000
    n_conditions = 5

    # Create condition labels
    condition_duration = n_timepoints // n_conditions
    conditions = np.repeat(np.arange(n_conditions), condition_duration)

    # Create data with distinct patterns per condition
    patterns = np.random.randn(n_conditions, n_features)
    data = np.zeros((n_features, n_timepoints))
    for i, cond in enumerate(conditions):
        data[:, i] = patterns[cond] + 0.1 * np.random.randn(n_features)

    # Create MVData object
    mvdata = MVData(data)

    print("Computing RDM from MVData object...")
    rdm, labels = rsa.compute_rdm_unified(mvdata, items=conditions)

    print(f"RDM shape: {rdm.shape}")
    print(f"Unique conditions: {labels}")

    # Visualize RDM
    fig = rsa.plot_rdm(
        rdm,
        labels=[f"Cond {i}" for i in labels],
        title="RDM from MVData",
        show_values=True,
    )
    plt.savefig("rsa_example_6.png")
    plt.close()

    return mvdata, rdm


if __name__ == "__main__":
    print("DRIADA RSA Examples")
    print("=" * 60)

    try:
        exp1, rdm1, labels1 = example_1_stimulus_conditions()
    except Exception as e:
        print(f"\nExample 1 skipped due to: {e}")

    try:
        similarity = example_2_compare_regions()
    except Exception as e:
        print(f"\nExample 2 skipped due to: {e}")

    try:
        similarity3 = example_3_compare_experiments()
    except Exception as e:
        print(f"\nExample 3 skipped due to: {e}")

    try:
        exp4, rdm4, labels4 = example_4_spatial_selectivity()
    except Exception as e:
        print(f"\nExample 4 skipped due to: {e}")

    try:
        rdm5a, rdm5b, sim5 = example_5_bootstrap_testing()
    except Exception as e:
        print(f"\nExample 5 skipped due to: {e}")

    try:
        mvdata6, rdm6 = example_6_mvdata_integration()
    except Exception as e:
        print(f"\nExample 6 skipped due to: {e}")

    print("\nAll examples completed!")
